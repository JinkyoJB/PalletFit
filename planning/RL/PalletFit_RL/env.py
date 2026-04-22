# planning/RL/PalletFit_RL/env.py
from __future__ import annotations
from dataclasses import dataclass, is_dataclass
from typing import Optional, Dict, Any, Deque, List, Tuple
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random, copy

from planning.data.trainset_generator import TrainsetGenerator
from planning.RL.PalletFit_RL.config import PREVIEW_MAX, ACTION_MAX_CANDIDATES, PIVOT_FEAT_DIM

from planning.RL.PalletFit_RL.reward_builder import build_reward
from planning.RL.PalletFit_RL.obs_builder import build_obs, make_obs_space_gym, queue_head
from planning.RL.PalletFit_RL.act_builder import place_by_action, rebuild_candidates

from planning.item import RotationType, Item
from planning.bin import Bin
from planning.BinSpecsDict import BIN_SPECS
from planning.packer import Packer

# ─────────────────────────────────────────────────────────────
# 랜덤 아이템 생성
# ─────────────────────────────────────────────────────────────
@dataclass
class TSGConfig:
    bin_size: tuple[int, int, int] = (1000, 1000, 1000)
    init_slice: tuple[int, int, int] = (4, 4, 1)
    margin_x: int = 0
    margin_y: int = 0
    min_item_mm: int = 100
    ensure_theoretical_100_su: bool = True

def _generate_items_with_tsg(tsg_cfg: TSGConfig) -> list[Item]:
    bx, by, bz = tsg_cfg.bin_size
    sx, sy, sz = tsg_cfg.init_slice

    def _shrink_for_min(total, slices, min_size):
        while slices > 1 and total // slices < min_size:
            slices -= 1
        return max(1, slices)

    sx = _shrink_for_min(bx, sx, tsg_cfg.min_item_mm)
    sy = _shrink_for_min(by, sy, tsg_cfg.min_item_mm)
    sz = _shrink_for_min(bz, sz, tsg_cfg.min_item_mm)

    gen = TrainsetGenerator(
        bin_size=[bx, by, bz],
        init_slice=[sx, sy, sz],
        margin_x=(0 if tsg_cfg.ensure_theoretical_100_su else tsg_cfg.margin_x),
        margin_y=(0 if tsg_cfg.ensure_theoretical_100_su else tsg_cfg.margin_y),
    )
    dict_items = gen.generate_trainset(num_merge=0, num_split=0)

    dict_items = [it for it in dict_items if min(it["width"], it["height"], it["depth"]) >= tsg_cfg.min_item_mm]

    if not dict_items:
        gen = TrainsetGenerator(
            bin_size=[bx, by, bz],
            init_slice=[1, 1, max(1, sz)],
            margin_x=0, margin_y=0,
        )
        dict_items = gen.generate_trainset(num_merge=0, num_split=0)

    items: list[Item] = []
    for i, d in enumerate(dict_items):
        d = d.copy()
        d.setdefault("priority", 7)
        d.setdefault("updown", False)
        d.setdefault("options", {"color": "#14ba5e"})
        d.setdefault("weight", 0)
        d.setdefault("loadbear", 0)
        d.setdefault("unit", "mm")
        d.setdefault("b_position", d.get("b_position", [-1, -1, -1]))
        d["name"] = d.get("name", f"tsg_{i}")
        d["partno"] = d.get("partno", str(i))
        d["objshape"] = "cube"
        d["rotation_quat"] = RotationType.RT_WHD if not isinstance(d.get("rotation_quat", 0), list) else d["rotation_quat"]
        items.append(Item(**d))
    return items

# ─────────────────────────────────────────────────────────────
# Gym Env
# ─────────────────────────────────────────────────────────────
class PalletFitEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, *, seed: int = 0, tb_log_dir: Optional[str | Path] = None, max_retry_per_step: int = 100):
        super().__init__()
        self._seed = int(seed)
        self._tb_log_dir = Path(tb_log_dir) if tb_log_dir else None

        # [수정] Config의 PREVIEW_MAX를 사용하여 차원 결정
        self._N_max = int(PREVIEW_MAX)
        self._K = int(ACTION_MAX_CANDIDATES)
        self._TOTAL = self._N_max * self._K
        self.NOOP_IDX = self._TOTAL - 1
        
        # ---------------- Observation / Action spaces ----------------
        base_obs_space = make_obs_space_gym()
        self.observation_space = spaces.Dict({
            **base_obs_space.spaces,
            "act_mask":  spaces.Box(0.0, 1.0, shape=(self._TOTAL,), dtype=np.float32),
            "act_cands": spaces.Box(0.0, 1.0, shape=(self._TOTAL, PIVOT_FEAT_DIM), dtype=np.float32),
        })
        self.max_steps_per_episode = 100
        
        # [수정] 통합된 변수 초기화 (기본값: Config상수)
        self._preview_cnt = int(PREVIEW_MAX)

        self.action_space = spaces.Discrete(self._TOTAL, start=0)

        # 내부 상태
        self._pending_plan: Optional[Dict[str, Any]] = None
        self._plan_in_use: Optional[Dict[str, Any]] = None

        self._episode_idx = 0
        self._steps_in_ep = 0
        self._ep_return = 0.0

        self._ep_reward_terms = defaultdict(float)   # 에피소드 누적

        self._last_reset_meta: Optional[Dict[str, Any]] = None
        self._head_indices: List[int] = []

        self.packer = None
        self.queue: Deque[int] = deque()

        # 액션 마스크 캐시
        self._last_cands: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None

        # 렌더 플래그
        self._save_render_on_done: bool = False
        self._save_render_dir: Optional[Path] = None

        # 시드 적용
        random.seed(self._seed)
        np.random.seed(self._seed)

        self.max_retry_per_step = max_retry_per_step
        self._current_step_retry_count = 0

    # ★ 추가: 관측/마스크를 NOOP-only로 만드는 안전 상태
    def _make_noop_only_state(self):
        self._last_mask = np.zeros((self._TOTAL,), dtype=bool)
        self._last_mask[self.NOOP_IDX] = True
        self._last_cands = np.zeros((self._TOTAL, PIVOT_FEAT_DIM), dtype=np.float32)

    # ── VecEnv/콜백에서 쓰는 헬퍼 ─────────────────────
    def get_SU(self) -> float:
        return float(self._bin.SU) if self._bin is not None else 0.0

    def pop_last_reset_meta(self):
        meta = self._last_reset_meta
        self._last_reset_meta = None
        return meta
    
    def apply_plan(self, plan: Any):
        """
        외부(agent)에서 플랜을 지정.
        - plan: dict 또는 EnvPlan (dataclass)
        - 다음 reset() 시점에 _pending_plan이 적용된다.
        """
        plan_dict = self._as_plan_dict(plan)
        self._pending_plan = plan_dict
        self._plan_in_use = plan_dict           # 현재 에피소드 info에 노출할 '사용 중 플랜'도 같이 갱신

    def _as_plan_dict(self, plan: Any) -> Dict[str, Any]:
        """
        dict 또는 EnvPlan → 통일된 dict 형식으로 변환
        """
        default = {
            "item_mode": "offline",
            "item_payload": {'offline_path':"planning/data/Item_data/exhibition/real_object_0331.json"},
            "bin": "experiment_RL",
            "bin_payload": {},
        }

        if plan is None:
            return dict(default)

        # dict인 경우
        if isinstance(plan, dict):
            item_mode = plan.get("item_mode",
                        plan.get("item_load_mode",
                        plan.get("mode", "offline")))

            item_payload = dict(plan.get("item_payload",
                                plan.get("payload", {})) or {})

            bin_key = plan.get("bin",
                    plan.get("bin_alias", "experiment_RL"))

            bin_payload = dict(plan.get("bin_payload", {}) or {})

            # [수정] EnvPlan 필드 -> Payload 매핑 (preview_cnt 하나만)
            alias_map = [
                ("bin_margin_x", "margin_x"),
                ("bin_margin_y", "margin_y"),
                ("preview_cnt", "preview_cnt"),  # <--- 통합된 변수 매핑
            ]
            for src, dst in alias_map:
                if src in plan and dst not in bin_payload:
                    bin_payload[dst] = plan[src]

            return {
                "item_mode": str(item_mode),
                "item_payload": item_payload,
                "bin": str(bin_key),
                "bin_payload": bin_payload,
            }

        # dataclass
        if is_dataclass(plan):
            from dataclasses import asdict
            return self._as_plan_dict(asdict(plan))

        return dict(default)


    # ── info 빌더 ────────────────────────────────────
    def _build_info(self, *, terminal_reason: Optional[str] = None) -> Dict[str, Any]:
        mode = self._plan_in_use.get("item_mode", "offline")
        ep_name = f"{mode}_ep{self._episode_idx}"
        
        # [수정] 통합된 변수 정보 기록
        preview_cnt = int(self._preview_cnt)
        
        SU_now = float(self.get_SU())
        packed_count = int(self._bin.size) if self._bin else 0

        info = {
            "episode_path": ep_name,
            "item_mode": mode,             
            "return": float(self._ep_return),
            "SU": SU_now,
            "packed_count": packed_count,
            "terminal_reason": terminal_reason or "",
            "plan": self._plan_in_use,
            "steps_in_ep": int(self._steps_in_ep),
            "preview_cnt": preview_cnt,
        }
        if getattr(self, "_ep_reward_terms", None):
            for k, v in self._ep_reward_terms.items():
                info[f"r_ep_{k}"] = float(v)
        return info

    # ── MaskablePPO용 액션 마스크 ────────────────────
    def action_masks(self) -> np.ndarray:   # 외부 호출용
        m = self._last_mask
        if m is None or m.size == 0 or not bool(m.any()):
            # 모두 불가능하면 최소한 NOOP 허용
            m = np.zeros((self._TOTAL,), dtype=bool)
            m[self.NOOP_IDX] = True
            self._last_mask = m
        return self._last_mask
    
    def save_render(self, save_dir: str | Path) -> None:
        base_dir = Path(save_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        self.packer.current_bin.render(
            save=True,
            save_path=str(base_dir),
            name=f"ep{self._episode_idx:04d}_step{self._steps_in_ep}_SU{self._bin.SU:.3f}",
            size_annotation=True
        )

    def _safe_rebuild_candidates(self, check: bool = False):
        try:
            # [수정] rebuild_candidates 호출 인자 간소화 (preview_cnt 전달)
            self._last_mask, \
            self._last_cands, \
            self._head_indices = rebuild_candidates(
                TOTAL=self._TOTAL,
                preview_cnt=self._preview_cnt, # <--- 통합된 변수
                queue=self.queue,
                bin=self._bin,
                K=self._K,
                NOOP_IDX=self.NOOP_IDX,
                cands_option='EDP',
                check=check
            )
        except Exception as e:
            print(f"[ERROR] _rebuild_candidates crashed: {e}")
            self._make_noop_only_state()
                
    # ── Gym API: reset/step ──────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        - self._pending_plan 이 있으면 그것을 사용하고,
        - 없으면 기존 self._plan_in_use 를 재사용,
        - 그것도 없으면 기본 플랜(offline, 빈 페이로드)을 사용한다.
        """
        self._episode_idx += 1
        self._steps_in_ep = 0
        self._ep_return = 0.0

        self._ep_reward_terms = defaultdict(float)
        if seed is not None:
            self._seed = int(seed)
            random.seed(self._seed)
            np.random.seed(self._seed)

        # ── 1) 이번 에피소드에 사용할 plan 확정 ─────────────
        if self._pending_plan is not None:
            plan = self._pending_plan
            self._pending_plan = None
            self._plan_in_use = plan
        elif self._plan_in_use is not None:
            plan = self._plan_in_use
        else:
            # 완전 초기 상태면 기본 플랜 생성
            plan = self._as_plan_dict(None)
            self._plan_in_use = plan

        # 평가 태그면 렌더 저장 켜기
        item_payload = plan.get("item_payload", {}) or {}
        tag = str(item_payload.get("tag", ""))
        if tag == "eval" and self._tb_log_dir is not None:
            self._save_render_on_done = True
            self._save_render_dir = self._tb_log_dir / "eval_renders"
        else:
            self._save_render_on_done = False
            self._save_render_dir = None

        if self._save_render_on_done and self._save_render_dir is not None:
            self._save_render_dir.mkdir(parents=True, exist_ok=True)

        # ── 2) bin 구성 ───────────────────────────────────────
        self.packer = self._build_binPacker_from_plan(plan)
        if self.packer is None:
            self.packer = Packer(rotation_type=RotationType.BasicRotation, order_setting=False)
            self.packer.build_bin(plan.get("bin", "experiment_RL"))

        # ── 3) 아이템 리스트 빌드 ─────────────────────────
        ok_items = self._build_items_from_plan(plan)        
        if not ok_items:
            # 아이템 빌드 실패: noop-only 상태로 빠짐
            self._safe_rebuild_candidates(check=True)
            head_q = queue_head(self._preview_cnt, self.queue)
            self._last_obs = build_obs(head_q, self._bin, self._last_mask, self._last_cands, self._TOTAL, self._preview_cnt)
            info = self._build_info(terminal_reason="invalid_item_build_failed")
            info["invalid_episode"] = True
            self._last_reset_meta = {
                "episode_idx": int(self._episode_idx),
                "episode_path": info["episode_path"],
            }
            return self._last_obs, info

        # ── 6) 정상 빌드 완료
        self._safe_rebuild_candidates(check=True)
        
        head_q = queue_head(self._preview_cnt, self.queue)
        obs = build_obs(head_q, self._bin, self._last_mask, self._last_cands, self._TOTAL, self._preview_cnt)
        info = self._build_info()
        self._last_reset_meta = {
            "episode_idx": int(self._episode_idx),
            "episode_path": info["episode_path"],
        }
        return obs, info

    def step(self, action: int):
        # ─────────────────────────────────────────────────────────
        # 1. 초기 체크 (타임아웃, 아이템 없음 등)
        # ─────────────────────────────────────────────────────────
        self._steps_in_ep += 1
        before_bin = copy.deepcopy(self._bin)

        # 타임아웃
        if self._steps_in_ep >= self.max_steps_per_episode:
            return self._finalize_step(
                terminated=False, truncated=True, terminal_reason="max_steps_reached", before_bin=before_bin
            )
        # 아이템 오링
        if len(self.queue) == 0:
             return self._finalize_step(terminated=True, truncated=False, terminal_reason="no_items", before_bin=before_bin)
        
        # ─────────────────────────────────────────────────────────
        # 2. 액션 실행 시도
        # ─────────────────────────────────────────────────────────
        if action == self.NOOP_IDX:
            # NOOP은 "포기" 선언이므로 종료 처리
            return self._finalize_step(
                terminated=True, truncated=False, terminal_reason="no_op_selected", before_bin=before_bin, failure_code="NOOP"
            )

        # 액션 시도
        ok, result = place_by_action(
            action, 
            self._preview_cnt, # <--- 통합된 변수
            self.queue, self._bin, 
            self._last_mask, self._last_cands, self._K
        )

        if ok:
            # ✅ 성공: 정상적으로 적재하고 다음 스텝으로 진행
            self._current_step_retry_count = 0  # [리셋]

            placed_item = result
            self.packer.bins[self.packer.current_bin_idx] = self._bin
            if placed_item is not None:
                try: self.queue.remove(placed_item._id)
                except ValueError: pass
            # self._bin.simplify()

            return self._finalize_step(
                terminated=False, truncated=False, terminal_reason=None, before_bin=before_bin
            )

        else:
            # ❌ 실패 (물리적 충돌 등): 죽이지 않고 "다시 해봐" 기회 제공
            error_code = result # 예: -3 (FAIL_COLLISION)
            
            # (1) 마스크 끄기
            self._last_mask[action] = False
            self._last_obs["act_mask"] = self._last_mask.astype(np.float32)
            
            # (2) 재시도 카운트 증가
            self._current_step_retry_count += 1
            
            # (3) 벌점 부여 (치명적이지 않게 -0.5 ~ -1.0 정도)
            # reward_builder를 호출하되, terminated=False로 호출하여 벌점만 가져옴
            step_penalty = -0.01  # 재시도 벌점 (너무 크면 아무것도 안 하려고 하니 작게)

            # (4) 종료 조건: 더 이상 갈 곳이 없거나 OR ★재시도 횟수 초과★
            all_masked = not np.any(self._last_mask[:self.NOOP_IDX])
            too_many_retries = self._current_step_retry_count >= self.max_retry_per_step

            if all_masked or too_many_retries:
                # 종료 시 리셋
                self._current_step_retry_count = 0
                
                reason = "all_actions_failed" if all_masked else "max_retries_exceeded"
                
                return self._finalize_step(
                    terminated=True, truncated=False, 
                    terminal_reason=reason,
                    before_bin=before_bin, 
                    failure_code="RETRY_LIMIT" # 실패 코드 전달
                )

            # (5) 재시도 상태 반환 (terminated=False)
            # info에 실패 사유를 적어주면 디버깅에 좋음
            info = self._build_info()
            info["retry_reason"] = f"Action {action} failed ({error_code})"
            
            return self._last_obs, step_penalty, False, False, info

    # ─────────────────────────────────────────────────────────
    # [헬퍼] 종료 처리 함수 (클래스 메서드로 변경)
    # ─────────────────────────────────────────────────────────
    def _finalize_step(self, *, terminated: bool, truncated: bool, terminal_reason: str | None, before_bin, failure_code=None):
        reward, terms = build_reward(
            before_bin=before_bin,
            after_bin=self._bin,
            terminated=terminated,
            truncated=truncated,
            finished=bool(terminal_reason in ["all_items_placed", "max_steps_reached"]),
            failure_code=failure_code
        )
        self._ep_return += reward
        if isinstance(terms, dict):
            for k, v in terms.items():
                self._ep_reward_terms[k] += float(v)

        # 다음 스텝을 위한 후보 생성 (check=False로 빠르게)
        self._safe_rebuild_candidates(check=True) 
        
        # 관측 빌드
        head_q = queue_head(self._preview_cnt, self.queue)        
        self._last_obs = build_obs(head_q, self._bin, self._last_mask, self._last_cands, self._TOTAL, self._preview_cnt)
        info = self._build_info(terminal_reason=terminal_reason or None)

        if (terminated or truncated) and getattr(self, "_save_render_on_done", False):
            try:
                save_dir = getattr(self, "_save_render_dir", None)
                if save_dir: Path(save_dir).mkdir(parents=True, exist_ok=True)
                self.save_render(save_dir)
            except: pass

        return self._last_obs, float(reward), bool(terminated), bool(truncated), info
    
    # ── 내부 유틸 ─────────────────────────────────────

    def _build_items_from_plan(self, plan) -> bool:
        if self.packer is None:
            return False

        item_payload = plan.get("item_payload", {}) or {}

        # 에피소드 시드 설정
        self._seed = int(item_payload.get("episode_seed", self._seed))
        
        # 모드 확인
        item_mode = plan.get("item_mode", "offline")

        try:
            # 1. 아이템 생성 또는 로드
            if item_mode == "tsg":
                tsg_cfg_dict = item_payload.get("tsg_cfg", {})
                tsg_cfg = TSGConfig(**tsg_cfg_dict)
                items = _generate_items_with_tsg(tsg_cfg)
                self.packer.items_list = items
                
            elif item_mode == "offline":
                paths = item_payload.get("offline_item_paths") or []
                path0 = random.choice(paths) if len(paths) > 0 else None
                if not path0:
                    return False
                self.packer.offline_item_path = path0
                self.packer._load_offline_data() # 내부에서 self.packer.items_list 채움
                
            elif item_mode == "online_type":
                paths = item_payload.get("online_item_type_path") or []
                path0 = random.choice(paths) if len(paths) > 0 else None
                if not path0:
                    return False
                self.packer.online_item_type_path = path0
                self.packer._load_online_item_type_data() # 내부에서 self.packer.items_list 채움

            # 3. Queue에는 아이템 객체나 인덱스가 아닌 'ID'만 저장
            self.queue = deque([it._id for it in self.packer.items_list])

            return len(self.queue) > 0

        except Exception as e:
            print(f"[WARN] _build_items_from_plan failed: {e}")
            self.packer.items_list = []
            self.queue = deque()
            return False


    def _build_binPacker_from_plan(self, plan) -> Packer:
        packer = Packer(rotation_type=RotationType.BasicRotation, order_setting=False)

        item_mode = plan.get("item_mode", "offline")
        if item_mode == "offline":
            bin_alias = "experiment_RL"
        elif item_mode == "online_type":
            bin_alias = "default2"
        else:
            bin_alias = plan.get("bin", "experiment_RL")

        bin_payload = plan.get("bin_payload", {}) or {}
        self.max_steps_per_episode = int(bin_payload.get("max_steps_per_episode", 100))
        
        # [수정] 통합된 변수 로드
        self._preview_cnt = int(bin_payload.get("preview_cnt", PREVIEW_MAX))

        if bin_alias in BIN_SPECS:
            packer.build_bin(bin_alias)
            packer.current_bin.margin_x = int(bin_payload.get("margin_x", 0))
            packer.current_bin.margin_y = int(bin_payload.get("margin_y", 0))
        else:
            print(f"[WARN] Unknown bin_alias '{bin_alias}'. Fallback to 'experiment_RL'.")
            packer.build_bin("experiment_RL")

        return packer

    @property
    def _bin(self) -> Optional[Bin]:
        if self.packer is None:
            return None
        return self.packer.current_bin

    @property
    def _items_list(self) -> List[Item]:
        if self.packer is None:
            return []
        return self.packer.items_list

    def render(self):
        return None

    def close(self):
        return super().close()
    
