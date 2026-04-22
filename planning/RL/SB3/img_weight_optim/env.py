# planning/RL/SB3/img_weight_optim/env.py
from __future__ import annotations
import gymnasium as gym, json, random, copy, time
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Optional
from PIL import Image

from planning.item                 import Item, RotationType
from planning.packer               import Packer
from planning.heuristics.PalletFit_v2 import W
from utils.util_functions import load_offline_data
# from planning.itemManager import global_item_manager

# ============================================================
# 전역 설정
# ============================================================

# 관측 이미지 크기
IMG_H, IMG_W, N_CH = 128, 128, 3

# 가중치 이름(고정 순서)
W_NAMES = [
    "ez_height_score","ez_cluster_score","guillotine",
    "match_area_sel","match_area_all","match_rA_all",
    "match_rB_all","weight_balance","bottomOverlapRatio",
]
DIM = len(W_NAMES)

# ✅ 절대 범위
WEIGHT_BOUNDS = {
    "ez_height_score":    (0.0, 50000.0),
    "ez_cluster_score":   (0.0, 20000.0),
    "guillotine":         (0.0, 1000.0),
    "match_area_sel":     (0.0, 10.0),
    "match_area_all":     (0.0, 10.0),
    "match_rA_all":       (0.0, 500.0),
    "match_rB_all":       (0.0, 500.0),
    "weight_balance":     (0.0, 100.0),
    "bottomOverlapRatio": (0.0, 10.0),
}
W_LO_VEC   = np.array([WEIGHT_BOUNDS[k][0] for k in W_NAMES], dtype=np.float32)
W_HI_VEC   = np.array([WEIGHT_BOUNDS[k][1] for k in W_NAMES], dtype=np.float32)
W_SPAN_VEC = np.maximum(W_HI_VEC - W_LO_VEC, 1e-6)  # 0 나눗셈 방지

# 보상(잠재함수) 관련
# su in [0,1],  Phi(s) = W_DELTA * su**SHAPE_P
SHAPE_P      = 0.5      # 0.5: 초반 채움 장려 / 2.0: 후반 미세개선 장려
W_DELTA      = 1.0      # 잠재함수 스케일
GAMMA_POT    = 0.999    # shaping용 감가율(모델의 gamma와 맞추는 것을 권장)
FAIL_PENALTY = 0.5      # 적재 실패 시 패널티(없애려면 0)
LAMBDA_NODE = 0.05  # 0.01~0.1에서 시작

# ===== Action scaling config =====
ACTION_SPACE_MODE = "normalized"   # "normalized" | "absolute"
ACTION_MAPPING    = "log1p"        # "linear" | "log1p" (권장: log1p)
ALPHA_POWER       = 0.5            # <1.0이면 상단 탐색 편향(초기 큰 값 더 잘 나옴)
BOUND_EPS         = 1e-3           # 경계 고착 방지 마진

USE_FAST2D  = False
FAST2D_MODE = "count"
GLOBAL_DEBUG_SAVE_EVERY = 0
# ============================================================



def packer_sampler():
    '''Packer 하나 생성'''
    pk = Packer(unfit_stop_setting=True,
                rotation_type=RotationType.BasicRotation,
                problem="online",
                model="PalletFit")
    pk.build_bin("experiment_RL")
    return pk



class ImgWeightOptimEnv(gym.Env):
    """
    - observation: {'image': (3,H,W) float32 in [0,1], 'feat': (4 + 1 + DIM,) float32}
    feat = [현재_아이템(4), 진행도(1), logw(DIM)]
    """
    metadata = {}

    def __init__(self, *,
                 render_every: int = 0,
                 item_seed: int | None = None,
                 log_dir: str = "planning/RL/SB3/img_weight_optim/logs/env",
                 item_mode: Literal["json", "episode_dir"] = "json",
                 episode_dir: str | Path | None = None,
                 episode_pattern: str = "dataset_episode_*.json",
                 item_json: str | Path = "$PALLETFIT_ROOT/planning/data/Item_data/paper/setting123_discrete/dataset_episode_001.json",
                 debug_img_dir: str | Path | None = "$PALLETFIT_ROOT/planning/RL/SB3/img_weight_optim/debug_img",
                 debug_save_every: int = GLOBAL_DEBUG_SAVE_EVERY,
                 worker_id: int = 0,
                 log_to_console: bool = True,
                 ):
        super().__init__()

        # RNG
        self.item_seed    = item_seed
        self.rng          = random.Random(item_seed) if item_seed is not None else random

        # 모드/데이터 소스
        self.render_every = int(render_every)
        self.item_mode    = item_mode
        self.item_json    = str(item_json)

        # 에피소드 파일 목록
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

        # Gym spaces
        if ACTION_SPACE_MODE == "normalized":
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(DIM,), dtype=np.float32)
        else:  # absolute 모드(기존 동작)
            self.action_space = gym.spaces.Box(low=W_LO_VEC, high=W_HI_VEC, shape=(DIM,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0.0, high=1.0, shape=(3, IMG_H, IMG_W), dtype=np.float32),
            "feat": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(4 + 1 + DIM,),   # next_item(4) + t_frac(1) + logw(DIM)
                dtype=np.float32
            )
        })

        # 로깅
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.worker_id      = worker_id
        self.log_to_console = bool(log_to_console)
        self._log_fp = open(Path(log_dir) / "stdout.log", "a", buffering=1, encoding="utf-8")

        # 디버그 이미지 경로
        self.debug_save_every = max(int(debug_save_every), 0)
        if debug_img_dir is not None and self.debug_save_every > 0:
            base_dir = Path(debug_img_dir)
            self.env_label = Path(log_dir).name  # ex) env_3
            self.debug_dir = base_dir / self.env_label
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.debug_dir = None
            self.env_label = Path(log_dir).name

        # 스텝/에피소드 카운터
        self._episode_idx = 0
        self._step        = 0
        self._global_step = 0
        self.last_episode_path: Optional[str] = None

        # === 절대 범위 → 벡터화 ===
        min_list, max_list = [], []
        for k in W_NAMES:
            lo, hi = WEIGHT_BOUNDS[k]
            lo, hi = float(lo), float(hi)
            if hi < lo:
                raise ValueError(f"{k}: min({lo}) > max({hi})")
            min_list.append(lo)
            max_list.append(hi)
        self.min_vec   = np.asarray(min_list, dtype=np.float32)
        self.max_vec   = np.asarray(max_list, dtype=np.float32)
        self.range_vec = np.maximum(self.max_vec - self.min_vec, 1e-8)

        # 내부 변환 변수
        self.cur_w = self.min_vec.copy()                 # boxed weights

        # 팔레트/휴리스틱
        self.bin = None
        self.packer  = None
        self._t0 = 0.0

        self._force_ep_path: Optional[Path] = None
        self._force_ep_index: Optional[int] = None

    def set_next_eval_episode(self, episode_path: Optional[str] = None, episode_index: Optional[int] = None):
        """다음 reset에서 사용할 에피소드를 강제 지정."""
        self._force_ep_path = Path(episode_path) if episode_path is not None else None
        self._force_ep_index = int(episode_index) if episode_index is not None else None

    def _pick_episode_file(self) -> Path:
        assert self.item_mode == "episode_dir", "episode_dir 모드에서만 사용"
        if self._force_ep_path is not None:
            p = self._force_ep_path
            # 한 번 쓰고 해제
            self._force_ep_path = None
            return p
        if self._force_ep_index is not None:
            files = self._episode_files  # 이미 sorted 되어 있음
            idx = self._force_ep_index % len(files)
            p = files[idx]
            self._force_ep_index = None
            return p
        # 기존 동작(랜덤)
        return self.rng.choice(self._episode_files)
    # --------------- 유틸 ---------------
    def _log(self, msg: str):
        if self.log_to_console:
            try:
                from tqdm.auto import tqdm
                tqdm.write(msg)
            except Exception:
                print(msg, flush=True)
        if self._log_fp:
            print(msg, file=self._log_fp, flush=True)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p) - np.log(1.0 - p)
    
    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)

        if ACTION_SPACE_MODE == "normalized":
            # [-1,1] -> [0,1]
            alpha = 0.5 * (np.clip(a, -1.0, 1.0) + 1.0)

            # 위쪽 편향: 0.5 -> sqrt(0.5)=0.707..., <1.0이면 큰 값 더 자주
            if ALPHA_POWER != 1.0:
                alpha = np.power(alpha, ALPHA_POWER)

            # 경계에서 살짝 안쪽으로
            alpha = alpha * (1.0 - 2.0 * BOUND_EPS) + BOUND_EPS

            if ACTION_MAPPING == "linear":
                w = self.min_vec + alpha * self.range_vec
            else:  # "log1p": zero-safe 기하 보간
                lo1p = np.log1p(self.min_vec)
                hi1p = np.log1p(self.max_vec)
                w = np.expm1(lo1p + alpha * (hi1p - lo1p))

            w = np.clip(w, self.min_vec, self.max_vec)

        else:
            # absolute 모드: 액션=가중치 그대로 사용(안전 clip만)
            w = np.clip(a, self.min_vec, self.max_vec)

        return w.astype(np.float32)

    # ------------------------------
    def _save_obs_png(self, arr: np.ndarray, tag: str):
        if self.debug_dir is None:
            return
        if self._global_step % self.debug_save_every != 0:
            return
        fname = f"{self.env_label}_g{self._global_step:06d}_ep{self._episode_idx:03d}_st{self._step:03d}_{tag}.png"
        Image.fromarray(arr).save(self.debug_dir / fname)

    def _save_arr2png(self, arr, fname: str):
        out_dir = Path("$PALLETFIT_ROOT/planning/RL/SB3/img_weight_optim/debug")
        out_dir.mkdir(parents=True, exist_ok=True)
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[0] in (1, 3, 4):  # CHW로 추정
            a = np.transpose(a, (1, 2, 0))           # HWC
        if a.dtype != np.uint8:
            if a.max() <= 1.0:
                a = (a * 255.0).clip(0, 255).astype(np.uint8)
            else:
                a = a.clip(0, 255).astype(np.uint8)
        Image.fromarray(a).save(out_dir / fname)

    # --------------- 관측 생성 ---------------
    def _obs_from_bin_fast2d(self, mode: str = FAST2D_MODE) -> np.ndarray:
        H, Wd = IMG_H, IMG_W
        occ = np.zeros((H, Wd), dtype=np.uint16)
        sx = Wd / float(self.bin.width)
        sy = H  / float(self.bin.depth)
        for item in self.bin.get_all_items():
            x, y, _ = item.b_position
            w, _, d = item.getDimension()
            x0 = max(0, min(Wd, int(x * sx)))
            x1 = max(0, min(Wd, int((x + w) * sx)))
            y0 = max(0, min(H,  int(y * sy)))
            y1 = max(0, min(H,  int((y + d) * sy)))
            if x1 > x0 and y1 > y0:
                occ[y0:y1, x0:x1] += 1
        occ_i = occ.astype(np.int32)
        if mode == "count":
            shade = np.clip(255 - 40 * occ_i, 0, 255).astype(np.uint8)
        elif mode == "alpha":
            shade = (255.0 * ((1.0 - 0.35) ** occ_i)).astype(np.uint8)
        else:
            shade = np.clip(255 - 40 * occ_i, 0, 255).astype(np.uint8)
        return np.repeat(shade[..., None], 3, axis=2)

    def _obs_from_bin(self) -> np.ndarray:
        if USE_FAST2D:
            arr = self._obs_from_bin_fast2d()
            self._save_obs_png(arr, tag="obs")
            return arr
        rgb = self.bin.render(show=False, save=False, return_array=True,
                              topdown=True, write_num=False)
        if rgb is None:
            rgb = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        im  = Image.fromarray(rgb).resize((IMG_W, IMG_H), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        self._save_obs_png(arr, tag="obs")
        return arr
    
    def _feat(self) -> np.ndarray:
        # 1) 현재 배치 대상 아이템 4D
        if self.item_idx < len(self.packer.items_list):
            it = self.packer.items_list[self.item_idx]
            item_to_place = np.array(
                [float(it.width), float(it.height), float(it.depth), float(it.weight)],
                dtype=np.float32
            )
        else:
            item_to_place = np.zeros(4, dtype=np.float32)

        # 2) 진행도 (0~1)
        t_frac = np.array([self.item_idx / max(1, len(self.packer.items_list))], dtype=np.float32)

        # 3) 현재 가중치 로그 정규화
        w_clamped = np.clip(self.cur_w, self.min_vec, self.max_vec)
        num = np.log1p(np.maximum(w_clamped - self.min_vec, 0.0))
        den = np.log1p(self.range_vec)
        logw = np.clip(num / den, 0.0, 1.0).astype(np.float32)

        return np.concatenate([item_to_place, t_frac, logw]).astype(np.float32)

    # --------------- Gym API ---------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 팔레트/휴리스틱 초기화
        self.item_idx = 0
        self.packer    = packer_sampler()
        self.bin      = self.packer.current_bin

        # 에피소드/아이템 소스
        if self.item_mode == "episode_dir":
            ep_path = self._pick_episode_file()   # ← 변경
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

        # 가중치 초기화: log-uniform in [min,max]
        u = np.random.random(DIM)  # in [0,1]
        self.cur_w = (W_LO_VEC + np.expm1(u * np.log1p(W_SPAN_VEC))).astype(np.float32)

        self._t0      = time.time()

        # PalletFit 전역 가중치 반영
        W.update({k: float(v) for k, v in zip(W_NAMES, self.cur_w)})

        obs = self._obs()
        self._episode_idx += 1
        self._step = 0

        self._log(f"\n── Episode {self._episode_idx} ─────────────────────────")
        if self.last_episode_path:
            self._log(f"   episode_file = {Path(self.last_episode_path).name}")

        return obs, {}

    def _obs(self) -> dict:
        raw_hwc = self._obs_from_bin().astype(np.uint8)   # (H,W,3) uint8
        self._last_final_img_hwc = raw_hwc.copy()         # 저장용
        img = raw_hwc.astype(np.float32) / 255.0          # 훈련용 (정규화)
        img_chw = np.transpose(img, (2, 0, 1))
        feat = self._feat().astype(np.float32)
        return {"image": img_chw, "feat": feat}
    
    # ------------------------------
    def step(self, action):
        self._step        += 1
        self._global_step += 1

        # ★ 액션 = 가중치 
        self.cur_w = self._action_to_weights(action)

        # PalletFit 전역 가중치 반영
        W.update({k: float(v) for k, v in zip(W_NAMES, self.cur_w)})

        # 현재 아이템 적재
        cur_item = copy.copy(self.packer.items_list[self.item_idx])
        prev_vol = self.bin.getTotalVolume()
        placed   = bool(self.packer.packingModel.stack(self.bin, [cur_item])[0])

        # SU 기반 잠재함수 shaping 보상 -----------------------------
        total_vol = self.bin.getTotalVolume()
        bin_vol   = max(1e-9, float(self.bin.volume))

        prev_su = prev_vol  / bin_vol *100.0
        su      = total_vol / bin_vol *100.0

        phi_prev = W_DELTA * (prev_su ** SHAPE_P)
        phi_cur  = W_DELTA * (su      ** SHAPE_P)

        shaped = GAMMA_POT * phi_cur - phi_prev

        graph_node_score = 0.0
        if placed:
            try:
                placed_id = self.bin.item_ids[-1]
                nb = self.bin.graph[placed_id]
                empty_cnt = int(len(nb['left'])==0) + int(len(nb['right'])==0) \
                        + int(len(nb['back'])==0) + int(len(nb['front'])==0)
                graph_node_score = LAMBDA_NODE * (empty_cnt / 4.0)
            except Exception:
                pass  # 그래프 접근 실패 시 보너스 0

        reward = shaped + graph_node_score

        if not placed:
            reward -= FAIL_PENALTY

        # 종료 조건
        if placed:
            self.item_idx += 1
            terminated = self.item_idx >= len(self.packer.items_list)
        else:
            terminated = True
        truncated = False

        # 로깅
        gs = self._global_step
        delta_su = (su - prev_su)
        self._log(
            f"[gStep {gs:6d} | eStep {self._step:3d}] placed={placed} | "
            f"reward={reward:6.3f} | SU={su:5.3f} | ΔSU={delta_su:5.3f} | "
            f"time={time.time()-self._t0:4.1f}s | weights={' '.join(f'{v:.2f}' for v in self.cur_w)}"
        )

        self.writer.add_scalar("reward/step", float(reward), gs)
        self.writer.add_scalar("su/current", float(su), gs)
        self.writer.add_scalar("su/delta", float(su - prev_su), gs)
        self.writer.add_scalar("reward/shaped", float(SHAPE_P), gs)
        self.writer.add_scalar("reward/graph_bonus", float(graph_node_score), gs)
        self.writer.add_scalar("reward/total", float(reward), gs)


        for k, v in zip(W_NAMES, self.cur_w):
            self.writer.add_scalar(f"W/{k}", float(v), gs)

        # 렌더(옵션)
        if self.render_every and gs % self.render_every == 0:
            self.bin.render(save=True, show=False,
                            save_path="planning/RL/SB3/img_weight_optim/snaps",
                            write_num=True,
                            name=f"g{gs:06d}_item{self.packer.items_list[self.item_idx].width}x{self.packer.items_list[self.item_idx].height}x{self.packer.items_list[self.item_idx].depth}_w{self.packer.items_list[self.item_idx].weight}")

        info = dict(
            n_placed=len(self.bin.item_ids),
            su=round(su, 3),
            delta_su=round(delta_su, 3),
        )
        if terminated or truncated:
            info["final_image_hwc"] = self._last_final_img_hwc  # 저장용 원본

        if self.last_episode_path:
            info["episode_file"] = Path(self.last_episode_path).name

        obs = self._obs()
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
