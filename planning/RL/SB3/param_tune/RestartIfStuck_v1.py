# planning/RL/SB3/param_tune/RestartIfStuck_v1.py
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np, torch

class RestartIfStuck(BaseCallback):
    """
    ▸ patience 회 연속으로 평가-reward 가 개선되지 않으면
      – policy 파라미터에 노이즈 주입(ε)
      – optimizer momentum 초기화
    """
    def __init__(self, eval_env, patience:int=5, eps:float=0.05):
        super().__init__()
        self.eval_env  = eval_env
        self.patience  = patience
        self.eps       = eps
        self.best_R    = -float("inf")
        self.no_up_cnt = 0

    # ────────────────────────────────────────
    def _on_step(self) -> bool:
        """필수: True 반환 → 학습 계속"""
        return True                      # 여기선 아무 일도 안 함

    # rollout(collect_rollouts) 이 끝날 때마다 호출
    def _on_rollout_end(self) -> None:
        obs, _ = self.eval_env.reset()
        ep_R   = 0.0
        while True:
            act, _ = self.model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = self.eval_env.step(act)
            ep_R += r
            if term or trunc:
                break

        # 개선 체크
        if ep_R > self.best_R + 1e-3:
            self.best_R    = ep_R
            self.no_up_cnt = 0
        else:
            self.no_up_cnt += 1

        # 정체이면 파라미터 흔들기
        if self.no_up_cnt >= self.patience:
            print(f"[Restart] reward 정체 → 파라미터 노이즈 주입")
            self.no_up_cnt = 0
            self._perturb_policy()

    # ────────────────────────────────────────
    def _perturb_policy(self):
        with torch.no_grad():
            for p in self.model.policy.parameters():
                p.add_(self.eps * torch.randn_like(p))
        # 옵티마이저 상태(모멘텀 등) 초기화
        self.model.policy.optimizer_state = {}
