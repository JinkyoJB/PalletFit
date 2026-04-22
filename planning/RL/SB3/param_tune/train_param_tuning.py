# planning/RL/SB3/param_tune/train_param_tuning.py
from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger   import configure

from planning.RL.SB3.param_tune.env_param_tuning_v1        import ParamTuneEnv
from planning.RL.SB3.param_tune.callbacks_param_tuning_v1  import SaveEvalCallback
from planning.RL.SB3.param_tune.RestartIfStuck_v1 import RestartIfStuck
import math

def cosine_lr_fn(base_lr: float, min_lr: float = 1e-5):
    """
    progress_remaining ∈ [1.0, 0.0]  ↦  lr
    - 처음 : base_lr
    - 끝   : min_lr
    """
    def _lr(progress_remaining: float) -> float:
        cos_out = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return min_lr + (base_lr - min_lr) * cos_out
    return _lr

# 사용 예시
lr_schedule   = cosine_lr_fn(base_lr=3e-4, min_lr=3e-5)
clip_schedule = cosine_lr_fn(base_lr=0.2 , min_lr=0.1)


# ───────────────────── 사용자 설정 ──────────────────────
ROOT                = Path("planning/RL/SB3/param_tune")
ITEM_SEED           = 42          # 학습·평가 공통 아이템 시퀀스
CKPT_PATH           = ROOT / "checkpoints" / "ckpt_00245760.zip"  # 시작점
TOTAL_ADDITIONAL_STEPS = 500_000*3  # 이어서 더 학습할 step 수

# 자원 활용 세팅
N_ENV       = 16      # 병렬 환경 수  (≈ CPU 코어 수와 유사하게)
N_STEPS     = 1024    # 1 Env 당 rollout 길이
BATCH_SIZE  = 512
N_EPOCHS    = 4
USE_SUBPROC = True    # 디버깅 시 False → DummyVecEnv
ENT_COEF      = 0.01          # <— 탐색 유지

policy_kwargs = dict(
    net_arch=[256,256,128],
    activation_fn=torch.nn.Tanh
)
# ───────────────────── 헬퍼 ────────────────────────────
def make_env(rank: int):
    """Subproc/Dummy VecEnv 가 요구하는 call-by-name wrapper"""
    return lambda: ParamTuneEnv(
        render_every=0,
        item_seed=ITEM_SEED,
        log_dir=ROOT / "logs" / f"env_{rank}"
    )

def load_or_create(model_env):
    tb_path = ROOT / "logs" / "ppo"
    logger  = configure(str(tb_path), ["stdout", "tensorboard"])

    if CKPT_PATH.is_file():
        print(f"[INFO] resume from {CKPT_PATH}")
        model = PPO.load(
            str(CKPT_PATH),
            env=model_env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=2,
        )
        model.set_logger(logger)

        # 하이퍼파라미터 덮어쓰기 (callable/정수 모두 OK)
        model.learning_rate = lr_schedule
        model.clip_range    = clip_schedule
        model.n_steps       = N_STEPS
        model.batch_size    = BATCH_SIZE
        model.n_epochs      = N_EPOCHS

    else:
        print("[INFO] fresh training")
        model = PPO(
            "MlpPolicy",
            model_env,
            learning_rate = lr_schedule,
            clip_range    = clip_schedule,
            n_steps       = N_STEPS,
            batch_size    = BATCH_SIZE,
            n_epochs      = N_EPOCHS,
            verbose       = 2,
            tensorboard_log = str(tb_path),
            ent_coef      = ENT_COEF,          # ★
            policy_kwargs = policy_kwargs,     # ★

            device        = "cuda" if torch.cuda.is_available() else "cpu",
        )
        model.set_logger(logger)

    return model

# ───────────────────── 메인 ────────────────────────────
if __name__ == "__main__":
    # CPU 쓰레드 제한 – env 프로세스에 코어를 더 남겨 줌
    torch.set_num_threads(8)
    os.environ["OMP_NUM_THREADS"] = "1"

    mp.set_start_method("fork", force=True)   # Windows 사용 시 주석

    # 1) 학습·평가 환경 --------------------------------------------------
    if USE_SUBPROC:
        train_env = SubprocVecEnv([make_env(i) for i in range(N_ENV)],
                                  start_method="fork")
    else:
        train_env = DummyVecEnv([make_env(i) for i in range(N_ENV)])

    eval_env  = ParamTuneEnv(render_every=0,
                             item_seed=ITEM_SEED,
                             log_dir=ROOT / "logs" / "eval_env")

    # 2) 모델 ------------------------------------------------------------
    model = load_or_create(train_env)
    already = model.num_timesteps
    target  = already + TOTAL_ADDITIONAL_STEPS

    # 3) 콜백 (체크포인트 + 평가) ----------------------------------------
    ROLL_OUT = N_STEPS * N_ENV           # 98_304
    SAVE_RATIO = 1                    # rollout의 1/4
    SAVE_EVERY_ROLLOUT = int(ROLL_OUT * SAVE_RATIO)  # 12_288

    cb_list = [
        SaveEvalCallback(eval_env, save_dir=ROOT/"checkpoints",
                        save_freq=SAVE_EVERY_ROLLOUT),
        RestartIfStuck(eval_env, patience=8, eps=0.03)   # 예: 8-rollout 동안 정체
    ]

    # 4) 학습 ------------------------------------------------------------
    print(f"[INFO] learn from {already} → {target} steps "
          f"({N_ENV} env × {N_STEPS} steps/rollout)")
    model.learn(
        total_timesteps=target,
        callback=cb_list,
        log_interval=1
    )

    # 5) 최종 저장 -------------------------------------------------------
    ROOT.mkdir(parents=True, exist_ok=True)
    final_path = ROOT / "pallet_param_final"
    model.save(final_path)
    print(f"[INFO] training done → {final_path}.zip")
