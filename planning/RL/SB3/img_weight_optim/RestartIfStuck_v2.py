# planning/RL/SB3/param_tune/RestartIfStuck.py
from __future__ import annotations
from collections import deque
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

def _is_vec_env(env) -> bool:
    # SB3 VecEnv들은 num_envs 속성을 가집니다.
    return hasattr(env, "num_envs")

def _rollout_one_episode(model, env):
    """eval_env에서 deterministic rollout 1회, (return, last_info) 반환"""
    is_vec = _is_vec_env(env)
    if is_vec:
        # VecEnv: reset() → obs (배치 포함)
        obs = env.reset()
        done = np.array([False] * env.num_envs)
        ep_R = 0.0
        last_info = {}
        while not np.any(done):
            # 관측값(dict)은 그대로 넣습니다. 절대 리스트로 감싸지 마세요.
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            # n_envs=1 가정: 첫 번째만 사용
            ep_R += float(rewards[0])
            done = dones
            last_info = infos[0]
        return ep_R, last_info
    else:
        # Raw gymnasium env: reset() → (obs, info)
        obs, _ = env.reset()
        ep_R = 0.0
        last_info = {}
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_R += float(reward)
            last_info = info
            if terminated or truncated:
                break
        return ep_R, last_info

class RestartIfStuck(BaseCallback):
    """
    최근 평가 리턴의 변화폭이 eps 미만으로 patience 번 연속이면
    '정체'로 보고 옵티마이저 상태를 리셋(가벼운 재시작)합니다.
    """
    def __init__(self, eval_env, patience: int = 8, eps: float = 0.03,
                 min_steps_between: int = 1000):
        super().__init__()
        self.eval_env = eval_env
        self.patience = int(patience)
        self.eps = float(eps)
        self.min_steps_between = int(min_steps_between)
        self.history = deque(maxlen=self.patience)
        self._last_check_step = 0

    def _on_rollout_end(self) -> bool:
        # 너무 자주 돌지 않게 조절(옵션)
        if self.num_timesteps - self._last_check_step < self.min_steps_between:
            return True
        self._last_check_step = self.num_timesteps

        ep_R, info = _rollout_one_episode(self.model, self.eval_env)
        self.history.append(ep_R)

        print(f"[StuckCheck @ {self.num_timesteps:8d}] "
              f"eval_return={ep_R:.3f} placed={info.get('n_placed','-')} "
              f"fill={info.get('fill','-')}")

        # 정체 감지: 최근 리턴 범위가 eps 미만
        if len(self.history) == self.history.maxlen:
            rng = float(max(self.history) - min(self.history))
            if rng < self.eps:
                print("[RestartIfStuck] Detected stagnation. Resetting optimizer state.")
                # 네트워크 가중치는 유지하고 옵티마이저 상태만 리셋
                opt = self.model.policy.optimizer
                opt.state = {}
                # (선택) 학습률 초기화
                if hasattr(self.model, "lr_schedule") and callable(self.model.lr_schedule):
                    for g in opt.param_groups:
                        g["lr"] = float(self.model.lr_schedule(1.0))
        return True
    
    # ★ 추가: 추상 메서드 구현(아무 작업도 하지 않음)
    def _on_step(self) -> bool:
        return True