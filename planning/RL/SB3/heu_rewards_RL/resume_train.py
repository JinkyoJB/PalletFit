# planning/RL/SB3/heu_rewards_RL/resume_train.py
import os
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure

from planning.RL.SB3.heu_rewards_RL.train import EpsMaskablePPO, make_env_thunk, FixedEpisodeEvalCallback, make_eval_vecenv  # 커스텀 클래스
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, ProgressBarCallback

# === 경로 설정 ===
CKPT     = "planning/RL/SB3/heu_rewards_RL/logs/final_model.zip"  # 이어서 학습할 모델
LOG_DIR  = "planning/RL/SB3/heu_rewards_RL/logs"
VECNORM  = f"{LOG_DIR}/vecnorm.pkl"                               # 학습 당시 VecNormalize 통계
SAVE_DIR = f"{LOG_DIR}/ckpt_resume"

EPISODE_DIR = "planning/data/Item_data/paper/setting123_discrete"

# === 환경 구성 파라미터 (원래 학습과 맞추는 걸 권장) ===
N_ENVS  = 16
SEED    = 84

def build_train_vecenv():
    """VecEnv + VecMonitor + VecNormalize 구성.
       기존 vecnorm 통계가 있으면 로드해서 동일 분포로 계속 학습."""
    vec = SubprocVecEnv([make_env_thunk(EPISODE_DIR, LOG_DIR, SEED, i) for i in range(N_ENVS)])
    vec = VecMonitor(vec)

    if os.path.exists(VECNORM):
        # 기존 정규화 통계 로드 (obs/ret의 평균, 분산 유지)
        vec = VecNormalize.load(VECNORM, vec)
        vec.training = True
        vec.norm_reward = True
    else:
        # 첫 재개이거나 파일이 없는 경우 — 새 정규화 시작
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        vec.training = True
        vec.norm_reward = True
    return vec

def main():
    assert os.path.exists(CKPT), f"Checkpoint not found: {CKPT}"
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    BEST_DIR = f"{LOG_DIR}/best"

    # 1) 학습용 VecEnv 구성 (정규화 포함)
    env = build_train_vecenv()

    # 2) 모델 로드 (하이퍼파라미터/스케줄 포함해서 복원됨)
    #    - env 를 반드시 전달해야 rollout 버퍼 등이 환경 크기에 맞게 재설정됨
    model: EpsMaskablePPO = EpsMaskablePPO.load(
        CKPT,
        env=env,
        device="auto",                     # GPU/CPU 자동 선택
        custom_objects={
            # 필요시 일부 하이퍼를 덮어쓸 수 있음(예: 학습률). 주석 해제해서 사용하세요.
            # "learning_rate": 1e-3,
            # "clip_range": 0.3,
            # "n_epochs": 5,
        },
        print_system_info=True,
    )

    # 3) 텐서보드 로거 확실히 붙이기
    tb_dir = f"{LOG_DIR}/tb"
    new_logger = configure(tb_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # 4) 콜백(체크포인트/프로그레스바) — 선택
    ckpt_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=SAVE_DIR,
        name_prefix="resume",
        save_replay_buffer=False,
    )

    FIXED_EPISODES = [
        "dataset_episode_619.json",
        "dataset_episode_1323.json",
        "dataset_episode_1001.json",
        "dataset_episode_1778.json",
        "dataset_episode_818.json",
        "dataset_episode_2249.json",
        "dataset_episode_621.json",
    ]
    # 평가용 별도 env (eval=True로 만들면 set_next_eval_episode가 활성 동작)
    EVAL_EP_PATHS = [str(Path(EPISODE_DIR) / name) for name in FIXED_EPISODES]

    eval_env = make_eval_vecenv(EPISODE_DIR)

    # 학습용 VecNormalize 통계 복사
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = getattr(env, "ret_rms", None)
    eval_env.training = False
    eval_env.norm_reward = False
    fixed_eval_cb = FixedEpisodeEvalCallback(
        eval_env=eval_env,
        episode_paths=EVAL_EP_PATHS,      # 또는 episode_indices=FIXED_EP_INDICES
        eval_freq=1_000,
        # eval_freq=1024,  # 디버그용,
        n_episodes=7,
        deterministic=True,
        best_model_save_path=BEST_DIR,
        tb_prefix="fixed_eval",
        verbose=1,
    )

    progress_cb = ProgressBarCallback()          # 진행률 바

    callbacks = CallbackList([
        progress_cb,
        ckpt_cb,
        fixed_eval_cb,
    ])

    # 5) 이어서 학습
    #    - total_timesteps는 "이번 세션 동안 더 돌릴 양"입니다. (기존 timesteps에 누적됨)
    MORE_STEPS = 900_000
    model.learn(total_timesteps=MORE_STEPS, callback=callbacks)

    # 6) 최신 VecNormalize 통계 & 모델 저장
    env.save(VECNORM)  # 계속 업데이트된 정규화 통계 저장
    model.save(os.path.join(LOG_DIR, "final_model_resumed.zip"))

    print("Done. Saved to:", os.path.join(LOG_DIR, "final_model_resumed.zip"))

if __name__ == "__main__":
    main()
