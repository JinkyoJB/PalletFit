# planning/RL/SB3/param_tune/env_param_tuning_v1.py
from __future__ import annotations
import gymnasium as gym, json, random, copy, time
from gym import spaces
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from planning.item                 import Item, RotationType
from planning.packer               import Packer
from planning.heuristics.PalletFit_v1 import PalletFit, LOWER_W, UPPER_W, CHANGE_DEPTH
from utils.get_value               import weight_balance_score

# ------------------------------------------------------------
_ITEM_JSON = "planning/data/Item_data/skt/demo_skt3.json"
PARAMS     = list(LOWER_W.keys()) + list(UPPER_W.keys()) + ["change_depth"]
DIM        = len(PARAMS)          # action-vector 길이

BASE_LOWER = np.array([LOWER_W[k] for k in LOWER_W], dtype=np.float32)
BASE_UPPER = np.array([UPPER_W[k] for k in UPPER_W], dtype=np.float32)
BASE_VEC   = np.concatenate([BASE_LOWER, BASE_UPPER, [CHANGE_DEPTH*1000]])

AMP        = 500.0
# ------------------------------------------------------------
def item_sampler(*, seed: int | None = None) -> list[Item]:
    """JSON ↦ Item 객체 목록 -- seed 고정 시 항상 같은 순서"""
    with open(_ITEM_JSON, encoding="utf-8") as fp:
        data = json.load(fp)
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(data)
    return [Item(**d) for d in data]

def bin_sampler():
    """빈 팔레트 하나 생성"""
    pk = Packer(unfit_stop_setting=True,
                rotation_type=RotationType.BasicRotation,
                problem="online")
    pk.build_bin("default1")
    return pk.current_bin
# ------------------------------------------------------------
class ParamTuneEnv(gym.Env):
    """
    - observation: dummy zero  
    - action     : 가중치 벡터 (연속)  
    - step       : 아이템 1개를 PalletFit 으로 배치
    """
    metadata = {}

    # ------------------------------
    def __init__(self, *,
                 render_every: int = 0,
                 item_seed: int | None = None,
                 log_dir: str = "planning/RL/SB3/param_tune/logs/env"):
        super().__init__()
        self.item_seed    = item_seed
        self.render_every = render_every

        self.action_space      = spaces.Box(-1.0, 1.0, shape=(DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # 카운터
        self._episode_idx = 0      # 에피소드 번호
        self._step        = 0      # 에피소드-step
        self._global_step = 0      # 전체-step

    # ------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.items     = item_sampler(seed=None)  # 새 아이템 세트
        self.item_idx  = 0                                  # 현재 아이템 인덱스
        self.bin       = bin_sampler()
        self._t0       = time.time()

        self._episode_idx += 1
        self._step = 0

        print(f"\n── Episode {self._episode_idx} ─────────────────────────")
        return np.zeros((1,), dtype=np.float32), {}

    # ------------------------------
    def step(self, action):
        self._step        += 1
        self._global_step += 1

        # 1) action → 실제 가중치 (0 ~ 1000)
        delta   = np.clip(action, -1.0, 1.0) * AMP
        w_real  = np.clip(BASE_VEC + delta, 0.0, 1000.0)

        pf     = PalletFit()
        pf.LOWER_W      = dict(zip(LOWER_W, w_real[:len(LOWER_W)]))
        pf.UPPER_W      = dict(zip(UPPER_W, w_real[len(LOWER_W):-1]))
        pf.CHANGE_DEPTH = np.clip(w_real[-1] / 1000.0, 0.2, 0.8)

        # 2) 현재 아이템 하나 적재
        cur_item = copy.copy(self.items[self.item_idx])
        placed   = bool(pf.stack(self.bin, [cur_item])[0])

        # 3) 보상 계산
        # reward_base = 1.0 if placed else 0.0
        fill_ratio  = self.bin.getTotalVolume() / self.bin.volume
        imbalance   = weight_balance_score(self.bin)
        reward      = 100.0 * fill_ratio - 0.1 * imbalance

        # 4) 종료 조건
        if placed:
            self.item_idx += 1
            terminated = self.item_idx >= len(self.items)   # 모든 아이템 소진
        else:
            terminated = True                               # 실패하면 바로 끝
        truncated = False

        # 5) 로그
        print(f"[gStep {self._global_step:6d} | eStep {self._step:3d}] "
              f"placed={placed} | reward={reward:6.3f} | "
              f"fill={fill_ratio:5.3f} | imb={imbalance:5.3f} | "
              f"time={time.time()-self._t0:4.1f}s")

        # 6) TensorBoard
        gs = self._global_step
        self.writer.add_scalar("reward", reward, gs)
        self.writer.add_scalar("fill_ratio", fill_ratio, gs)
        self.writer.add_scalar("imbalance", imbalance, gs)

        for k, v in zip(PARAMS, w_real):
            self.writer.add_scalar(f"weight/{k}", v, gs)

        # 7) 렌더 (옵션)
        if self.render_every and gs % self.render_every == 0:
            self.bin.render(save=True, show=False,
                            save_path="planning/RL/SB3/param_tune/snaps",
                            write_num=True,
                            name=f"g{gs:06d}")
            for k, v in zip(PARAMS, w_real):
                self.writer.add_scalar(f"action/{k}", v, gs)

        info = dict(n_placed=len(self.bin.item_ids),
                    fill=round(fill_ratio, 3),
                    imbalance=round(imbalance, 3))
        return np.zeros((1,), dtype=np.float32), reward, terminated, truncated, info

    # ------------------------------
    def close(self):
        self.writer.close()
        super().close()
