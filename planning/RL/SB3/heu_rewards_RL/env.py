# planning/RL/SB3/heu_rewards_RL/env.py
from __future__ import annotations
import gymnasium as gym
import json, random, copy, time, os
from pathlib import Path
from typing import Optional, Literal
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from planning.item import Item, RotationType
from planning.packer import Packer
from utils.Pivot import Pivot

from utils.checkPivot import (
    checkPivot_R, 
    SUCCESS,
    FAIL_OUT_OF_BOUNDS_NEG, FAIL_OUT_OF_BOUNDS_POS,
    FAIL_COLLISION, FAIL_WEIGHT_EXCEEDED, FAIL_NO_TOP_EMPTY,
    FAIL_NO_SUPPORT_BOTTOM, FAIL_SUPPORT_OVERLOAD, FAIL_OVERHANG_TOO_MUCH,
    FAIL_SUPPORT_AREA_INSUFFICIENT, FAIL_CG_OUTSIDE_SUPPORT, FAIL_CUMULATIVE_UNSTABLE,
)
from utils.pivot_generation import (
    project_lines_left_to_pivots,
    project_lines_front_to_pivots,
    project_lines_down_to_pivots,
    project_lines_down_to_pivots2left,
    project_lines_down_to_pivots2front,
    get_pivots_ep,
    get_pivots_cp,
    get_pivots_ems
    
) 
from utils.get_value import (
    get_direction_overlap,
    score_ez_distribution,
    get_score_Guillotine,
    balance_score,
)
from utils.util_functions import load_offline_data

from planning.RL.SB3.heu_rewards_RL.image_views import MultiViewRenderer

# ───────── 보상 하이퍼 (기존 shaping 계열과 유사) ─────────
FAIL_BASE    = 0.1     # 기본 실패 패널티 스케일

# ───────── obs : image관련 하이퍼 ─────────
IMG_H, IMG_W, N_CH = 128, 128, 3
DEFAULT_LOG_ROOT   = Path("planning/RL/SB3/heu_rewards_RL/logs")
DEFAULT_DEBUG_ROOT = Path("planning/RL/SB3/heu_rewards_RL/debug_img")

A_SU = 5.0  # 안정성 보상 스케일

# ───────── 헬퍼 함수 ─────────
def packer_sampler():
    pk = Packer(
        rotation_type=RotationType.BasicRotation,
        problem="online",
        model="PalletFit",
    )
    pk.build_bin("experiment_RL")
    return pk

def _to_snake(name: str) -> str:
    import re as _re
    return _re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


# ───────── Env ─────────
class ImgHeurObsRewardEnv(gym.Env):

    FAIL_LIST = [
        FAIL_OUT_OF_BOUNDS_NEG, FAIL_OUT_OF_BOUNDS_POS, FAIL_COLLISION, FAIL_WEIGHT_EXCEEDED,
        FAIL_NO_TOP_EMPTY, FAIL_NO_SUPPORT_BOTTOM, FAIL_SUPPORT_OVERLOAD, FAIL_OVERHANG_TOO_MUCH,
        FAIL_SUPPORT_AREA_INSUFFICIENT, FAIL_CG_OUTSIDE_SUPPORT, FAIL_CUMULATIVE_UNSTABLE
    ]
    metadata = {}

    def __init__(
        self,
        *,
        render_every: int = 0,
        item_seed: int | None = None,
        log_dir: str | Path | None = None,
        item_mode: Literal["json", "episode_dir"] = "episode_dir",
        episode_dir: str | Path | None = None,
        episode_pattern: str = "dataset_episode_*.json",
        item_json: str | Path = "planning/data/Item_data/skt/demo_skt3.json",
        debug_img_dir: str | Path | None = None,
        debug_save_every: int = 0,
        is_eval_env: bool = False,
        worker_id: int = 0,
        log_to_console: bool = True,
        use_fast2d: bool = False,
        fast2d_mode: Literal["count", "alpha"] = "alpha",
    ):
        super().__init__()
        self.item_seed = item_seed
        self.rng = random.Random(item_seed) if item_seed is not None else random
        self.render_every = int(render_every)
        self.item_mode = item_mode
        self.item_json = str(item_json)
        self.is_eval_env = bool(is_eval_env)
        self._force_ep_path: Optional[Path] = None
        self._force_ep_index: Optional[int] = None

        cls_tag = _to_snake(self.__class__.__name__)
        role    = "eval" if self.is_eval_env else "train"
        self.env_label = f"{cls_tag}_{role}_{int(worker_id)}"

        if log_dir is not None:
            root = Path(log_dir)    # 사용자가 준 log_dir를 '루트'로 보고, 충돌 방지를 위해 env_label 서브폴더를 자동으로 붙임
        else:
            root = DEFAULT_LOG_ROOT

        self.log_dir = root / self.env_label
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if debug_img_dir is None and debug_save_every > 0:
            self.debug_dir = (DEFAULT_DEBUG_ROOT / cls_tag / self.env_label)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        elif isinstance(debug_img_dir, (str, Path)) and debug_save_every > 0:
            self.debug_dir = Path(debug_img_dir); self.debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.debug_dir = None

        self.writer = SummaryWriter(str(self.log_dir), max_queue=10, flush_secs=5, filename_suffix=f".{os.getpid()}")
        self.worker_id = worker_id
        self.log_to_console = bool(log_to_console)
        self._log_fp = open(self.log_dir / "stdout.log", "a", buffering=1, encoding="utf-8")
        self.debug_save_every = max(int(debug_save_every), 0)

        self._episode_files: list[Path] = []
        if self.item_mode == "episode_dir":
            if episode_dir is None:
                raise ValueError("item_mode='episode_dir'인데 episode_dir가 없습니다.")
            ep_dir = Path(episode_dir).expanduser()
            if not ep_dir.is_dir():
                raise FileNotFoundError(f"에피소드 디렉토리를 찾을 수 없습니다: {ep_dir}")
            self._episode_files = sorted(ep_dir.glob(episode_pattern))
            if not self._episode_files:
                raise RuntimeError(f"에피소드 파일이 없습니다: {ep_dir} / 패턴={episode_pattern}")

        self.views = MultiViewRenderer(img_h=IMG_H, img_w=IMG_W, use_fast2d=use_fast2d, fast2d_mode=fast2d_mode)

        self._episode_idx = 0
        self._step = 0
        self._global_step = 0
        self.last_episode_path: Optional[str] = None

        self.packer = None
        self.bin = None
        self.item_idx = 0
        self._t0 = 0.0
        self._placed_count = 0

        self.FEAS_MASK = 512  # = 1536

        self.observation_space = gym.spaces.Dict({
            # 멀티뷰 이미지
            "image_top":   gym.spaces.Box(0, 255, shape=(3, IMG_H, IMG_W), dtype=np.uint8),
            # "image_left":  gym.spaces.Box(0, 255, shape=(3, IMG_H, IMG_W), dtype=np.uint8),
            # "image_front": gym.spaces.Box(0, 255, shape=(3, IMG_H, IMG_W), dtype=np.uint8),

            # 현재 아이템 피처 [w,h,d,wt,t_frac]
            "feat": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            # 좌표는 bin 크기로 정규화해서 [0,1] (무효 후보는 0으로 패딩)
            "cand_pos":  gym.spaces.Box(low=0.0, high=1.0, shape=(self.FEAS_MASK, 4), dtype=np.float32),  # (x,y,z, rot_id(0/1))
            # 후보 유효 여부 마스크(정책에도 힌트로 제공; 액션 마스크는 별도 Wrapper로 전달)
            "cand_mask":   gym.spaces.Box(low=0.0, high=1.0, shape=(self.FEAS_MASK,),   dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(self.FEAS_MASK) # 0 ~ FEAS_MASK-1

        # 후보 캐시
        self._cand_cache_pos  = np.zeros((self.FEAS_MASK, 4), dtype=np.float32)  # (x,y,z, rt_norm)
        self._cand_cache_mask = np.zeros((self.FEAS_MASK),    dtype=np.float32)  # 0/1
        self.strict_mask = True  # 성공 가능한 후보만 마스크로 쓸지 여부(원하면 False로 꺼도 됨)
        self._cand_cache_feas = np.zeros((self.FEAS_MASK), dtype=np.float32)  # SUCCESS 가능한 후보 마스크
        self._cand_index_map  = {}  # (d,k) -> (px,py,pz, rt_obj)

        # ── 보상 통계(EMA) 초기화: 평균, 제곱평균(msq) ──
        # 분산 = msq - mean^2. 초기 분산=1.0로 시작해 급폭 방지
        self._ezh_mean = 0.0; self._ezh_msq = 1.0  # height
        # self._ezc_mean = 0.0; self._ezc_msq = 1.0  # cluster
        self._gu_mean  = 0.0; self._gu_msq  = 1.0
        self._ba_mean  = 0.0; self._ba_msq  = 1.0

    # ---------- eval hooks ----------
    def set_next_eval_episode(self, episode_path: Optional[str] = None, episode_index: Optional[int] = None):
        if not self.is_eval_env:
            return
        self._force_ep_path = Path(episode_path) if episode_path is not None else None
        self._force_ep_index = int(episode_index) if episode_index is not None else None

    def _pick_episode_file(self) -> Path:
        assert self.item_mode == "episode_dir"
        if self.is_eval_env and (self._force_ep_path is not None or self._force_ep_index is not None):
            if self._force_ep_path is not None:
                p = self._force_ep_path; self._force_ep_path = None; return p
            idx = self._force_ep_index % len(self._episode_files)
            self._force_ep_index = None
            return self._episode_files[idx]
        return self.rng.choice(self._episode_files)
    
    def _dbg_scan_candidates(self, item) -> dict:
        """
        현재 캐시된 모든 후보(flat index)들을 검사해
        성공(SUCCESS) 개수와 실패코드 히스토그램을 반환.
        반환: {"ok": int, "hist": {code:int -> count:int}}
        """
        ok = 0
        hist: dict[int, int] = {}

        # self._cand_index_map: flat_k -> (px,py,pz,rot_id)  # (또는 + src_dir)
        for _k, entry in self._cand_index_map.items():
            # entry 형태 호환( (px,py,pz,rot_id) 또는 (px,py,pz,rot_id, src_dir) )
            if len(entry) == 5:
                px, py, pz, rot_id, _ = entry
            else:
                px, py, pz, rot_id = entry

            q = self._rot_id01_to_quat(rot_id)
            code, _ = checkPivot_R(self.bin, copy.deepcopy(item), [px, py, pz], q, apply_margin=True)

            c = int(code)
            hist[c] = hist.get(c, 0) + 1
            if code == SUCCESS:
                ok += 1

        return {"ok": ok, "hist": hist}


    # ---------- logging ----------
    def _log(self, msg: str):
        if self.log_to_console:
            try:
                from tqdm.auto import tqdm
                tqdm.write(msg)
            except Exception:
                print(msg, flush=True)
        if self._log_fp:
            print(msg, file=self._log_fp, flush=True)

    def _save_png(self, arr: np.ndarray, tag: str):
        if self.debug_dir is None or self.debug_save_every <= 0:
            return
        if self._global_step % self.debug_save_every != 0:
            return
        from PIL import Image
        # 현재 배치 대상 아이템 크기 붙이기
        dims = self._get_pending_item_dims()
        dims_str = f"_{self._fmt_dims_whd(dims)}" if dims is not None else ""
        fname = (
            f"{self.env_label}"
            f"_g{self._global_step:06d}"
            f"_ep{self._episode_idx:03d}"
            f"_st{self._step:03d}"
            f"_{tag}"
            f"{dims_str}.png"
        )
        Image.fromarray(arr).save(self.debug_dir / fname)

    def _save_views_triplet(self, tag: str):
        """현재 MultiViewRenderer 상태를 (top/left/front)로 저장."""
        if self.debug_dir is None or self.debug_save_every <= 0:
            return
        try:
            top_hwc, left_hwc, front_hwc = self.views.get_triplet_hwc()
            self._save_png(top_hwc,   f"{tag}_top")
            self._save_png(left_hwc,  f"{tag}_left")
            self._save_png(front_hwc, f"{tag}_front")
        except Exception as e:
            self._log(f"[debug] _save_views_triplet failed: {e}")


    # ───────── 추가: 현재 아이템 크기 얻기 + 포맷 ─────────
    def _get_pending_item_dims(self) -> Optional[tuple[float, float, float]]:
        """지금 스텝에서 '배치하려는' 아이템의 (w,h,d) 반환. 없으면 None."""
        try:
            if self.packer is None:
                return None
            if not (0 <= self.item_idx < len(self.packer.items_list)):
                return None
            it = self.packer.items_list[self.item_idx]
            return float(it.width), float(it.height), float(it.depth)
        except Exception:
            return None

    @staticmethod
    def _fmt_dims_whd(whd: tuple[float, float, float]) -> str:
        """(w,h,d)를 파일명용 문자열로. 정수면 .0 제거, 실수면 소수 2자리까지."""
        def _fmt(x: float) -> str:
            xi = int(round(x))
            return str(xi) if abs(x - xi) < 1e-6 else f"{x:.2f}".rstrip("0").rstrip(".")
        w, h, d = whd
        return f"item{_fmt(w)}x{_fmt(h)}x{_fmt(d)}"
    
    # --- Rotation 0/1 <-> Quaternion helper --------------------------------
    @staticmethod
    def _quat_is(q, ref, atol=1e-3):
        try:
            import numpy as _np
            return _np.allclose(_np.asarray(q, dtype=_np.float32),
                                _np.asarray(ref, dtype=_np.float32), atol=atol)
        except Exception:
            return q == ref  # 최후의 보루(정확도↓)

    def _quat_to_rot_id01(self, q):
        """
        q가 RT_WHD이면 0, RT_HWD이면 1, 그 외는 None.
        """
        if q is None:
            return None
        if self._quat_is(q, RotationType.RT_WHD):
            return 0
        if self._quat_is(q, RotationType.RT_HWD):
            return 1
        return None
    
    def _rot_id01_to_quat(self, rid):
        """
        0 -> RT_WHD, 1 -> RT_HWD
        """
        return RotationType.RT_WHD if int(rid) == 0 else RotationType.RT_HWD
            
    # ----------action -------------
    def valid_action_mask(self) -> np.ndarray:
        # strict면 feasible, 아니면 관측 마스크 사용
        base = self._cand_cache_feas if self.strict_mask else self._cand_cache_mask
        mask = (base > 0.5).astype(bool) if base is not None else np.zeros(self.K_PER_DIR, dtype=bool)

        # 1차 폴백: feasible이 전부 0이면 관측 마스크로
        if not mask.any():
            mask = (self._cand_cache_mask > 0.5)

        # 2차 폴백: 그래도 전부 0이면 최소 1개 True 강제
        if not mask.any():
            mask = np.zeros(self.K_PER_DIR, dtype=bool)
            if len(self._cand_index_map) > 0:
                k0 = next(iter(self._cand_index_map.keys()))
                mask[k0] = True
            else:
                mask[0] = True  # 정말 아무 후보도 없을 때 완전 비상용

        return mask

    # ----------reward ----------
    # def _ema_norm(self, x: float, mean_attr: str, msq_attr: str, beta: float = EMA_BETA) -> float:
    #     """
    #     x를 EMA(mean, msq)로 업데이트하고 표준화해서 반환.
    #     s = sqrt(max(msq - mean^2, 1e-6))
    #     """
    #     m = getattr(self, mean_attr)
    #     q = getattr(self, msq_attr)

    #     # EMA 업데이트 (mean, mean of squares)
    #     m = (1.0 - beta) * m + beta * x
    #     q = (1.0 - beta) * q + beta * (x * x)

    #     var = max(q - m * m, 1e-6)
    #     s = float(np.sqrt(var))

    #     setattr(self, mean_attr, m)
    #     setattr(self, msq_attr, q)

    #     return (x - m) / s
    
    # ---------- obs ----------
    def _feat(self) -> np.ndarray:
        if self.item_idx < len(self.packer.items_list):
            it = self.packer.items_list[self.item_idx]
            w,h,d = float(it.width), float(it.height), float(it.depth)
            wt    = float(it.weight)
        else:
            w=h=d=wt=0.0
        t_frac = (self.item_idx+1) / max(1, len(self.packer.items_list))
        return np.array([w,h,d,wt,t_frac], dtype=np.float32)



    def _obs(self) -> dict:
        # ① 후보 캐시 최신화
        self._build_candidate_cache()

        # ② 멀티뷰 이미지 (반환 포맷이 무엇이든 안전하게 (3,H,W) uint8로 강제)
        top, left, front = self.views.get_triplet_hwc()  # 이름이 hwc여도 실제는 chw일 수 있음
        top   = self.views.ensure_chw_u8(top)
        # left  = self.views.ensure_chw_u8(left)
        # front = self.views.ensure_chw_u8(front)

        self._last_final_img_hwc = self.views.chw_to_hwc_u8(top).copy()

        # 디버그 저장시 HWC u8로
        # self._save_png(self.views.chw_to_hwc_u8(left),  "obs_left")
        self._save_png(self.views.chw_to_hwc_u8(top),   "obs_top")
        # self._save_png(self.views.chw_to_hwc_u8(front), "obs_front")
        
        # 관측으로 내보낼 마스크 선택
        mask_for_obs = self._cand_cache_feas if self.strict_mask else self._cand_cache_mask

        obs = {
            "image_top":   top,        # (3,H,W) uint8
            # "image_left":  left,
            # "image_front": front,
            "feat":        self._feat(),
            "cand_pos":    self._cand_cache_pos.copy(),
            "cand_mask":   mask_for_obs.copy(),
        }

        # 최종 방어선: space 일치 확인
        assert obs["image_top"].dtype == np.uint8 and obs["image_top"].shape == (3, IMG_H, IMG_W)
        return obs

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.item_idx = 0
        self.packer = packer_sampler()
        self.bin = self.packer.current_bin
        self.views.attach_bin(self.bin)
        self._placed_count = 0

        if self.item_mode == "episode_dir":
            ep_path = self._pick_episode_file()
            item_data = load_offline_data(ep_path)
            for it in item_data:
                self.packer.addItem(Item(**it))
            self.last_episode_path = str(ep_path)
        else:
            with open(self.item_json, encoding="utf-8") as fp:
                data = json.load(fp)
            self.rng.shuffle(data)
            for d in data:
                self.packer.addItem(Item(**d))
            self.last_episode_path = None

        self._t0 = time.time()
        self._episode_idx += 1
        self._step = 0

        self._log(f"\n── (HEUR) Episode {self._episode_idx} ─────────────────────────")
        if self.last_episode_path:
            self._log(f"   episode_file = {Path(self.last_episode_path).name}")

        obs, info = self._obs(), {}
        return obs, info
    
    def _build_candidate_cache(self):
        K = self.FEAS_MASK
        self._cand_cache_pos.fill(0.0)
        self._cand_cache_mask.fill(0.0)
        self._cand_cache_feas.fill(0.0)
        self._cand_index_map.clear()

        bw, bh, bd = float(self.bin.width), float(self.bin.height), float(self.bin.depth)
        eps = 1e-9

        # 1) 두 방향 후보 생성
        piv_front = self._get_pivots_for_dir("front")
        piv_left  = self._get_pivots_for_dir("left")

        # 2) front + left 를 순서 유지하며 합치되, (x,y,z,rot) 기준으로 중복 제거
        def _rid(pv):
            return self._quat_to_rot_id01(getattr(pv, "rt", None))
        def _key(pv):
            r = _rid(pv)
            if r is None:
                return None
            return (
                round(float(pv.x), 3),
                round(float(pv.y), 3),
                round(float(pv.z), 3),
                int(r),
            )
        seen = set()
        merged = []
        for pv in (piv_front + piv_left):
            k = _key(pv)
            if k is None or k in seen:
                continue
            seen.add(k)
            merged.append(pv)

        # 3) K개 한도까지 채움
        fill = 0
        for pv in merged:
            if fill >= K:
                break
            rid = _rid(pv)              # 0/1
            px, py, pz = float(pv.x), float(pv.y), float(pv.z)

            # [0,1] 정규화
            nx = 0.0 if bw <= eps else np.clip(px / bw, 0.0, 1.0)
            ny = 0.0 if bh <= eps else np.clip(py / bh, 0.0, 1.0)
            nz = 0.0 if bd <= eps else np.clip(pz / bd, 0.0, 1.0)

            self._cand_cache_pos[fill, 0] = nx
            self._cand_cache_pos[fill, 1] = ny
            self._cand_cache_pos[fill, 2] = nz
            self._cand_cache_pos[fill, 3] = float(rid)

            self._cand_cache_mask[fill] = 1.0
            self._cand_index_map[fill]  = (px, py, pz, int(rid))
            fill += 1

        # 4) feasible 마스크 (strict_mask일 때만)
        if self.strict_mask and (0 <= self.item_idx < len(self.packer.items_list)):
            base_item = self.packer.items_list[self.item_idx]
            for k, (px, py, pz, rot_id) in self._cand_index_map.items():
                q = self._rot_id01_to_quat(rot_id)
                code, _ = checkPivot_R(self.bin, copy.deepcopy(base_item), [px, py, pz], q, apply_margin=True)
                if code == SUCCESS:
                    self._cand_cache_feas[k] = 1.0

            # 전부 0이면 폴백
            if not (self._cand_cache_feas > 0.5).any():
                self._cand_cache_feas[...] = self._cand_cache_mask

        # 모든 후보가 없으면 (아주 드묾) 더미 pivot 두 개 추가
        if len(self._cand_index_map) == 0:
            for rid in (0, 1):
                k = rid  # 0, 1
                if k >= self.K_PER_DIR: break
                self._cand_cache_pos[k, 0] = 0.0
                self._cand_cache_pos[k, 1] = 0.0
                self._cand_cache_pos[k, 2] = 0.0
                self._cand_cache_pos[k, 3] = float(rid)
                self._cand_cache_mask[k] = 1.0
                self._cand_index_map[k] = (0.0, 0.0, 0.0, rid)

    def _get_pivots_for_dir(self, dir_name: str):
        """
        dir_name ∈ {"front", "left"}
        - 여러 projection 결과를 합치고 (priority 순서 유지)
        - (x,y,z,rt)를 기준으로 안정적으로 중복 제거
        """
        def _dedup_keep_order(seq):
            uniq, seen = [], set()
            for pv in seq:
                rid = self._quat_to_rot_id01(getattr(pv, "rt", None))
                if rid is None:
                    continue  # WHD/HWD 아닌 회전은 사용하지 않음
                key = (
                    round(float(pv.x), 3),
                    round(float(pv.y), 3),
                    round(float(pv.z), 3),
                    int(rid),
                )
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(pv)
            return uniq
        
        # 빈 박스일 경우 원점 좌표기준 후보 2개만 반환
        if self.bin.origin_item_id is None:
            return [Pivot( 0, 0, 0, RotationType.RT_WHD, bench_bin=self.bin), Pivot( 0, 0, 0, RotationType.RT_HWD, bench_bin=self.bin)]

        if dir_name == "front":
            lst = []
            lst.extend(project_lines_front_to_pivots(self.bin))        # 1차: front
            cands_d = project_lines_down_to_pivots(self.bin)           # 공통 down (재사용)
            lst.extend(cands_d)                                        # 2차: down
            lst.extend(project_lines_down_to_pivots2front(self.bin, cands_d))  # 3차: down→front
            return _dedup_keep_order(lst)

        elif dir_name == "left":
            lst = []
            lst.extend(project_lines_left_to_pivots(self.bin))         # 1차: left
            cands_d = project_lines_down_to_pivots(self.bin)           # 공통 down (재사용)
            lst.extend(cands_d)                                        # 2차: down
            lst.extend(project_lines_down_to_pivots2left(self.bin, cands_d))   # 3차: down→left
            return _dedup_keep_order(lst)

        return []
    
    # ----------- step() 구현 관련 메서드 -----------

    def step(self, action):
        """
        (x,y,z,rot) 제안 → checkPivot_R로 검증 → 성공 시 커밋, 실패 시 코드별 패널티.
        보조헤드 타깃으로 실패코드 one-hot을 info에 넣어줌.
        """
        self._global_step += 1
        self._step += 1

        # 기본 반환값
        terminated = False
        truncated  = False
        info = {}

        # 모든 아이템 소진 시 종료
        if self.item_idx >= len(self.packer.items_list):
            return self._obs(), 0.0, True, False, info

        # 현재 아이템(원본 보호)
        cur_item = copy.deepcopy(self.packer.items_list[self.item_idx])

        # ── 배치 전: 보조 점수들 (delta 계산용) ─────────────────
        def _safe(f, default=0.0):
            try:
                return float(f())
            except Exception:
                return float(default)
            
        # --- step() 위쪽 헬퍼에 추가 (또는 step 내부에 정의해도 OK)
        def _safe_tuple(f, default):
            try:
                return f()
            except Exception:
                return default

        # ez_prev_h, ez_prev_c = _safe_tuple(lambda: score_ez_distribution(self.bin), (0.0, 0.0))
        # ez_prev_h, _ = _safe_tuple(lambda: score_ez_distribution(self.bin), (0.0, 0.0))

        # guill_prev = _safe(lambda: get_score_Guillotine(self.bin), 0.0)
        # bal_prev   = _safe(lambda: balance_score(self.bin), 0.0)

        # ---------- 액션 디코딩 ----------
        k = int(action)
        valid = (0 <= k < self.FEAS_MASK) and (self._cand_cache_mask[k] > 0.5)

        if not valid:
            # 무효 선택: 작은 패널티 주고 종료(혹은 계속 진행하도록 설계 가능)
            return self._obs(), -0.1, True, False, {"placed": False, "invalid_choice": True}

        # 매핑된 실제 피봇 사용
        px, py, pz, rot_id = self._cand_index_map[k]
        quat = self._rot_id01_to_quat(rot_id)

        # 배치 가능성 체크
        code, candidate = checkPivot_R(self.bin, cur_item, [px, py, pz], quat, apply_margin=True)

        info["pivot_code"] = int(code)
        info["pivot_pos"]  = [px, py, pz]
        info["rot_idx"]    = int(rot_id)        # 0/1 보고

        # 원하면 쿼터니언도 같이 로그
        # info["rot_quat"] = quat

        # ---------- 성공 처리 ----------
        if code == SUCCESS:
            self.bin.store(candidate)
            try:
                self.views.update_with_item(candidate)
            except Exception:
                self.views.rebuild_from_bin()
            self.item_idx += 1

            # reward +=1.0  # 성공 보너스

            reward += float(self.bin.SU) * A_SU  # SU 가중치

            # (6) 로깅 업데이트 (이전/이후/델타/정규화 모두 남김)
            info.update({
                "placed": True,
                "su": round(self.bin.SU, 3),
                "R_terms": {
                    "SU_RM": float(self.bin.SU) * A_SU,
                    "reward_total": reward,
                }
            })
            # info["aux_reg"] = np.array([d_ezh, d_ezc, d_gu, d_bal, reward], dtype=np.float32)
            try:
                gs = self._global_step
                self.writer.add_scalar("reward/SU",              float(self.bin.SU) * A_SU, gs)
                self.writer.add_scalar("reward/SU_Reward",      float(self.bin.SU) * A_SU, gs)
                self.writer.add_scalar("reward/total",           reward,       gs)
            except Exception:
                pass

            if self.item_idx >= len(self.packer.items_list):
                terminated = True

        # ---------- 실패 처리 ----------
        else:
            t_frac = (self.item_idx+1) / max(1, len(self.packer.items_list))  # 0~1
            reward = -FAIL_BASE * (1.0 + 0.5 * (1.0 - t_frac))  # 실패 패널티 (진행률 반영)

            info.update({"placed": False, "su": float(self.bin.SU)})
            info["cand_scan"] = self._dbg_scan_candidates(cur_item)
            # 이전 버전과 유사하게 에피소드 종료로 처리 (원하면 '아이템 스킵'도 가능)
            terminated = True

        # ---------- (옵션) 스냅샷 ----------
        if self.render_every and (self._global_step % self.render_every == 0):
            try:
                dims = self._get_pending_item_dims()
                dims_str = self._fmt_dims_whd(dims) if dims is not None else "item0x0x0"
                self.bin.render(
                    save=True, show=False,
                    save_path="planning/RL/SB3/heu_rewards_RL/snaps",
                    write_num=True,
                    name=f"g{self._global_step:06d}_{dims_str}"
                )
            except Exception:
                pass

        # ---------- 관측 반환 ----------
        obs = self._obs()
        scan_str = ""
        scan = info.get("cand_scan")
        if scan is not None:
            ok = int(scan.get("ok", 0))
            hist = scan.get("hist", {})
            # 코드:개수 를 정렬해서 한 줄로
            hist_str = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
            scan_str = f" | cand_ok={ok} | cand_hist={{ {hist_str} }}"
        
        n_valid = int((self._cand_cache_mask > 0.5).sum())
        scan_str += f" | valid={n_valid}"
        
        self._log(
            f"[env {self.env_label}] Step {self._step} "
            f"| Action (k={k}) "
            f"| Placed: {info.get('placed', False)} "
            f"| SU: {info.get('su', 0.0):.3f} "
            f"| Reward: {reward:.3f} "
            f"{scan_str} "                                  # ← 여기서 cand_scan 추가
            f"| Time: {time.time() - self._t0:.1f}s"
        )
        return obs, float(reward), terminated, truncated, info


    def close(self):
        try:
            if getattr(self, "_log_fp", None):
                self._log_fp.close()
                self._log_fp = None
        except Exception:
            pass
        self.writer.close()
        super().close()
