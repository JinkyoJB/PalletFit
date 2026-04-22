# planning/RL/PalletFit_RL/rl_adapter.py
from __future__ import annotations

from collections import deque
from itertools import islice
from typing import Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch as th
from torch.distributions import Distribution
from sb3_contrib.ppo_mask import MaskablePPO

# PyTorch 2.11+에서 validate_args=True가 기본값으로 바뀌어 MaskableCategorical에서
# Simplex 검증 오류가 발생할 수 있음 → 학습 환경과 동일하게 False로 설정
Distribution._validate_args = False

from planning.bin import Bin
from planning.item import Item
# [공통 모듈 Import]
from planning.RL.PalletFit_RL.obs_builder import build_observation_deque, queue_head
from planning.RL.PalletFit_RL.act_builder import rebuild_candidates, place_by_action
from planning.RL.PalletFit_RL.config import (
    ACTION_MAX_CANDIDATES,
    PREVIEW_MAX,      # [수정] 이제 PREVIEW_MAX를 기준으로 동작
    PIVOT_FEAT_DIM,
)
from planning.itemManager import global_item_manager

if TYPE_CHECKING:
    from planning.packer import Packer


class PalletFitRLAdapter:
    """
    Packer용 PalletFit-RL 어댑터.
    env.py와 동일한 builder 함수들을 사용하여 로직 일관성을 유지합니다.
    """

    def __init__(
        self,
        *,
        packer: "Packer",
        weight_file_dir: str,
        selectable_len: int = 5,
        unfit_stop_setting: bool = True,
        rotation_type=None,
        device: Optional[str] = None,
        cands_option: str = 'EDP'  # 'EDP', 'CP', 'EP', 'EMS'
    ) -> None:
        self.packer: "Packer" = packer
        self.weight_file_dir: str = weight_file_dir
        self.unfit_stop_setting: bool = unfit_stop_setting
        self.rotation_type = rotation_type
        self.cands_option = cands_option

        # [수정] RL action space 설정 (PREVIEW_MAX 기준)
        self._N_max: int = int(PREVIEW_MAX)
        self._K: int = int(ACTION_MAX_CANDIDATES)
        self._TOTAL: int = self._N_max * self._K
        self.NOOP_IDX: int = self._TOTAL - 1
        
        # [수정] 가변 설정 통합 (preview_cnt)
        # 사용자가 요청한 selectable_len을 preview_cnt로 설정하되, 최대값 제한
        self._preview_cnt: int = int(min(selectable_len, PREVIEW_MAX))

        # 디바이스 설정
        if device is None:
            device = "cuda" if th.cuda.is_available() else "cpu"
        self.device: str = device

        # 모델 로드
        self.model: MaskablePPO = MaskablePPO.load(
            self.weight_file_dir,
            device=self.device,
        )

        # 상태 캐시
        self._queue: Deque[int] = deque()
        self._head_indices: List[int] = [] # rebuild_candidates에서 갱신됨
        self._last_mask: Optional[np.ndarray] = None
        self._last_cands: Optional[np.ndarray] = None

    def stack(self, bin_obj: Bin, items_list: List[Item]) -> List[int]:
        """
        [수정됨] 단일 스텝 예측:
        items_list 중 가장 적절한 아이템 1개를 선택하여 적재하고 즉시 종료한다.
        - 성공한 아이템 인덱스: 1
        - 나머지 아이템: 0
        """
        n_items = len(items_list)
        # 결과 배열 초기화 (0으로 설정)
        fit_result: List[int] = [0] * n_items

        if bin_obj is None or n_items == 0:
            return fit_result

        # 1. Queue 초기화 (Item ID로 관리)
        self._queue = deque([item._id for item in items_list])
        
        # ID -> Index 매핑 (결과 반환용)
        id_to_idx = {item._id: i for i, item in enumerate(items_list)}

        # 2. 후보 갱신 (Action Mask & Candidates 생성)
        # [수정] 변경된 rebuild_candidates 시그니처 사용 (preview_cnt)
        self._last_mask, self._last_cands, self._head_indices = rebuild_candidates(
            TOTAL=self._TOTAL,
            preview_cnt=self._preview_cnt, # <--- 통합된 변수 전달
            queue=self._queue,
            bin=bin_obj,
            K=self._K,
            NOOP_IDX=self.NOOP_IDX,
            cands_option=self.cands_option, 
            check=True 
        )

        # 3. 유효한 액션이 있는지 확인 (NOOP 제외)
        if not np.any(self._last_mask[:self.NOOP_IDX]):
            # 적재 가능한 곳이 없으면 모두 0 반환하고 종료
            return fit_result

        # 4. 관측 생성
        # [수정] queue_head에 preview_cnt 전달
        head_q = queue_head(self._preview_cnt, self._queue)
        
        # [수정] build_observation_deque에 preview_cnt 전달
        raw_obs = build_observation_deque(
            bin_obj=bin_obj,
            queue=head_q,
            preview_cnt=self._preview_cnt
        )
        
        # 마스크/후보 추가
        obs = {
            **raw_obs,
            "act_mask": self._last_mask.astype(np.float32),
            "act_cands": self._last_cands.astype(np.float32),
        }

        # NaN/Inf 방어: observation에 비정상 값이 있으면 0으로 교체
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and not np.isfinite(v).all():
                obs[k] = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)

        # 5. RL 추론
        with th.no_grad():
            action, _ = self.model.predict(
                obs,
                deterministic=True,
                action_masks=self._last_mask
            )
            action = int(action)

        # 6. NOOP 체크
        if action == self.NOOP_IDX:
            return fit_result

        # 7. 액션 실행
        # [수정] place_by_action 시그니처 변경 (preview_cnt)
        ok, placed_item = place_by_action(
            action=action,
            preview_cnt=self._preview_cnt, # <--- 통합된 변수 전달
            queue=self._queue,
            bin_obj=bin_obj,
            last_mask=self._last_mask,
            last_cands=self._last_cands,
            K=self._K
        )

        if ok:
            # 성공 시 해당 아이템 인덱스만 1로 마킹
            if placed_item._id in id_to_idx:
                idx = id_to_idx[placed_item._id]
                fit_result[idx] = 1
        
        self.packer.bins[self.packer.current_bin_idx] = bin_obj      
        return fit_result