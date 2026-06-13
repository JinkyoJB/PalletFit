# planning/RL/SB3/img_weight_optim/train.py
from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecNormalize

from planning.RL.SB3.img_weight_optim.env        import ImgWeightOptimEnv
from planning.RL.SB3.img_weight_optim.callbacks  import SaveTopKCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from planning.RL.SB3.param_tune.callbacks_epoch_announce import EpochAnnounceCallback
import math
from sb3_contrib import RecurrentPPO
from utils.env_paths import repo_path


def cosine_lr_fn(base_lr: float, min_lr: float = 1e-5):
    """
    progress_remaining ∈ [1.0, 0.0]  ↦  lr
    - 처음 : base_lr
    - 끝   : min_lr
    """
    def _lr(progress_remaining: float) -> float:
        cos_out = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        # print(f"progress_remaining: {progress_remaining:.6f}, cos_out: {cos_out:.6f}")
        return min_lr + (base_lr - min_lr) * cos_out
    return _lr

# learning_rate   = cosine_lr_fn(base_lr=3e-4, min_lr=3e-5)
# clip_range = cosine_lr_fn(base_lr=0.2 , min_lr=0.1)
learning_rate=3e-4
clip_range=0.2

# ───────────────────── 사용자 설정 ──────────────────────
ROOT                = Path("planning/RL/SB3/img_weight_optim")

ITEM_SEED           = 42
USE_EPISODE_DIR     = True
EPISODE_DIR         = repo_path("planning", "data", "Item_data", "paper", "setting123_discrete")
ITEM_JSON_PATH      = "planning/data/Item_data/skt/demo_skt3.json"

CKPT_PATH           = ROOT / "checkpoints" / "topk_su_step01425408_metric48_700000.zip"

# 자원 활용 세팅
N_ENV       = 16      # 병렬 환경 수  (≈ CPU 코어 수와 유사하게)
N_STEPS     = 128    # 1 Env 당 rollout 길이
ROLL_OUT = N_STEPS * N_ENV           # 16 × 64 = 1_024 스텝마다 업데이트
n_mini_batches = 4               # 1 업데이트 당 미니배치 
BATCH_SIZE  = ROLL_OUT // n_mini_batches  # 512
N_EPOCHS    = 3               # 1 업데이트 당 epoch 수
ADDITIONAL_STEPS = 10_000  # 추가 학습 스텝 수
TARGET_KL = 0.0125

SAVE_RATIO = 3
SAVE_EVERY_ROLLOUT = int(ROLL_OUT * SAVE_RATIO) # 10_240 롤아웃마다 저장/평가

USE_SUBPROC = True
ENT_COEF      = 0.01          # <— 탐색 유지

policy_kwargs = dict(
    lstm_hidden_size=256,
    n_lstm_layers=1,
    shared_lstm=True,        # 공유 LSTM 사용
    enable_critic_lstm=False,# ★ 공유를 쓰면 반드시 False로 명시
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[256, 128], vf=[256, 128]),
    normalize_images=False,  # 이미 env에서 정규화
)

# ───────────────────── 헬퍼 ────────────────────────────
def make_env(rank: int):
    return lambda: Monitor(
        ImgWeightOptimEnv(
            render_every=10_000,
            item_seed=ITEM_SEED + rank,
            log_dir=ROOT / "logs" / f"env_{rank}",
            item_mode="episode_dir" if USE_EPISODE_DIR else "json",
            episode_dir=str(EPISODE_DIR) if USE_EPISODE_DIR else None,
            item_json=ITEM_JSON_PATH,
            debug_img_dir=None,
            worker_id=rank,
            log_to_console=(rank == 0),  # 0번 워커만 콘솔 출력
        ),
        filename=None
    )


def make_eval_env():
    return Monitor(
        ImgWeightOptimEnv(
            render_every=0,
            item_seed=ITEM_SEED,   # 재현성
            log_dir=ROOT / "logs" / "eval_env",
            item_mode="episode_dir" if USE_EPISODE_DIR else "json",
            episode_dir=str(EPISODE_DIR) if USE_EPISODE_DIR else None,
            item_json=ITEM_JSON_PATH,
            debug_img_dir=None,
        ),
        filename=None
    )

def load_or_create(model_env):
    tb_path = ROOT / "logs" / "ppo"

    # if CKPT_PATH.is_file():
    #     print(f"[INFO] resume from {CKPT_PATH}")
    #     model = RecurrentPPO.load(
    #         CKPT_PATH,
    #         env=model_env,
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         print_system_info=True,
    #     )

    #     from stable_baselines3.common.logger import configure
    #     logger = configure(str(tb_path), ["stdout", "tensorboard"])
    #     model.set_logger(logger)

    #     # 하이퍼파라미터 덮어쓰기 (callable/정수 모두 OK)
    #     model.learning_rate = learning_rate
    #     model.clip_range    = clip_range
    #     model.n_steps       = N_STEPS
    #     model.batch_size    = BATCH_SIZE
    #     model.n_epochs      = N_EPOCHS

    if CKPT_PATH.is_file():
        print(f"[INFO] resume from {CKPT_PATH}")
        model = RecurrentPPO.load(
            CKPT_PATH,
            env=model_env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            print_system_info=True,
        )

        from stable_baselines3.common.logger import configure
        logger = configure(str(tb_path), ["stdout", "tensorboard"])
        model.set_logger(logger)

        # (1) 스케줄로 덮어쓰기  ← 여기 핵심!
        model.lr_schedule = get_schedule_fn(learning_rate)  # float → schedule
        model.clip_range  = get_schedule_fn(clip_range)     # float → schedule
        # (clip_range_vf 쓰면 model.clip_range_vf = get_schedule_fn(…))

        # (2) 학습 크기 하이퍼 갱신
        # 있으면 세팅(없으면 경고만)
        if hasattr(model, "target_kl"):
            model.target_kl = TARGET_KL     # ★ 추가
        else:
            print("[WARN] this RecurrentPPO build has no `target_kl` attr")
        model.n_steps    = N_STEPS
        model.batch_size = BATCH_SIZE
        model.n_epochs   = N_EPOCHS

        # (3) 버퍼/내부 상태 재구성 (env 붙여서 다시 세팅)
        model.set_env(model_env)   # ← rollout_buffer를 새 n_steps/n_env에 맞게 재생성

    else:
        print("[INFO] fresh training")
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            model_env,
            policy_kwargs=policy_kwargs,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=0.999,
            learning_rate=learning_rate,
            clip_range=clip_range,
            ent_coef=ENT_COEF,
            target_kl=TARGET_KL,           # ★ 추가
            verbose=2,
            tensorboard_log=str(tb_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


        # ↓ fresh에서도 명시적으로 붙여주면 더 확실
        from stable_baselines3.common.logger import configure
        logger = configure(str(tb_path), ["stdout", "tensorboard"])
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

    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = DummyVecEnv([make_eval_env])          # ← VecEnv로 감싸기
    # 2) 모델 ------------------------------------------------------------
    model = load_or_create(train_env)

    # 2-1) 추가 학습 스텝(rollout 배수로 반올림)
    if ROLL_OUT % BATCH_SIZE != 0:
        raise ValueError(
            f"BATCH_SIZE({BATCH_SIZE})는 ROLL_OUT({ROLL_OUT})의 약수여야 합니다."
        )

    # 추가 스텝을 rollout 배수로 반올림(위로)
    iters                       = (ADDITIONAL_STEPS + ROLL_OUT - 1) // ROLL_OUT
    ADDITIONAL_STEPS_ROUNDED    = iters * ROLL_OUT

    already = int(model.num_timesteps)
    target  = already + ADDITIONAL_STEPS_ROUNDED

    print(
        f"[INFO] additional={ADDITIONAL_STEPS:,} → rounded={ADDITIONAL_STEPS_ROUNDED:,} "
        f"(rollout={ROLL_OUT:,}) | already={already:,} → target={target:,}"
    )
    updates = ADDITIONAL_STEPS_ROUNDED // ROLL_OUT
    minibatches_per_update = ROLL_OUT // BATCH_SIZE
    print(f"[INFO] planned updates={updates:,} | minibatches/update={minibatches_per_update} | "
        f"epochs/update={N_EPOCHS}")

    # 3) 콜백 (체크포인트 + 진행바 등) ----------------------------------------

    cb_list = [
        ProgressBarCallback(),
        EpochAnnounceCallback(),
        SaveTopKCallback(
            eval_env,
            save_dir=ROOT / "checkpoints",
            save_freq=SAVE_EVERY_ROLLOUT,  # 예: 5×rollout마다
            monitor="su",                # 또는 "return"
            mode="max",
            top_k=3,                       # ★ 상위 3개만 유지
            save_eval_snap=True,           # 각 top-k에 대응하는 PNG도 저장
            write_param_hist=True,         # 파라미터 히스토그램/L2 기록
        ),
    ]


    # 4) 학습 ------------------------------------------------------------
    if ADDITIONAL_STEPS_ROUNDED > 0:
        model.learn(
            total_timesteps=target,        # ← 누적 기준 목표치(이어달리기)
            callback=cb_list,
            log_interval=1,
            reset_num_timesteps=False,     # ← 스텝 카운터/스케줄 유지
        )
    else:
        print("[INFO] ADDITIONAL_STEPS_ROUNDED == 0, 학습을 건너뜁니다.")


    # 5) 최종 저장 -------------------------------------------------------
    ROOT.mkdir(parents=True, exist_ok=True)
    import time
    date = time.strftime("%y%m%d")
    final_path = ROOT / f"{date}_final"
    model.save(final_path)
    print(f"[INFO] training done → {final_path}.zip")