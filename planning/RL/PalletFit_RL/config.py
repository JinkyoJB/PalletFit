# planning/RL/PalletFit_RL/config.py
from __future__ import annotations

# ── Observation ─────────────────────────────────────────────
OBS_TOPK_DEFAULT: int = 64
ITEM_FEAT_DIM: int = 22     # encode_item() 결과 길이
# [SU, height, cluster, guillotine_squashed, imbalance_norm, I_norm]
GLOBAL_FEAT_DIM: int = 6

# ── Action (pivots) ────────────────────────────────────────
ACTION_MAX_CANDIDATES: int = 360    # action mask 최대 길이
PIVOT_FEAT_DIM: int = 4            # [x/W, y/D, z/H, rot_id]
DEDUP_ROUND_NDIGITS: int = 3
ALLOWED_ROT_IDS = [0, 1]           # 허용 회전 id (RotationType.quat_to_index 결과)
PREVIEW_MAX = 10               # action 후보 미리보기 개수의 최대값, 바꾸면 모델 재학습 필요

# ── Reward ─────────────────────────────────────────────────
REWARD_MODE_DEFAULT: str = "delta"     # {'delta', 'absolute', 'delta_clip0'}
SUPPORTED_REWARD_MODES = {"delta", "absolute", "delta_clip0"}