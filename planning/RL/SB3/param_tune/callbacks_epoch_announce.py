# planning/RL/SB3/param_tune/callbacks_epoch_announce.py
from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback

class EpochAnnounceCallback(BaseCallback):
    """
    각 rollout이 끝나고 PPO.train()에 들어가기 직전에
    '이번 업데이트에서 n_epochs 번 도는 중'을 콘솔에 찍어줍니다.
    업데이트가 끝난 뒤에는 요약을 한 줄로 찍습니다.
    """
    def __init__(self):
        super().__init__()
        self._last_update_step = 0
        
    def _on_step(self) -> bool:
        # 여기서는 아무 것도 하지 않음. 매 스텝마다 True만 반환.
        return True

    def _on_rollout_end(self) -> None:
        # 여기 시점이 곧 PPO.train() 들어가기 직전
        m = self.model
        try:
            # callable일 수도 있으니 현재 값으로 평가
            clip = m.clip_range(m._current_progress_remaining) if callable(m.clip_range) else m.clip_range
            lr   = m.learning_rate(m._current_progress_remaining) if callable(m.learning_rate) else m.learning_rate
        except Exception:
            clip, lr = m.clip_range, m.learning_rate

        total_mb = m.rollout_buffer.buffer_size // m.batch_size
        print(
            f"[Update start @ timesteps={m.num_timesteps}] "
            f"epochs={m.n_epochs}, batch_size={m.batch_size}, "
            f"mini_batches/update={total_mb}, lr={lr:.3g}, clip={clip:.3g}"
        )
        self._last_update_step = m.num_timesteps

    def _on_training_end(self) -> None:
        # learn() 전체 끝에서 한 번
        print(f"[Training finished] total_timesteps={self.model.num_timesteps}")

    def _on_rollout_start(self) -> None:
        # 업데이트 직후 다음 rollout이 시작될 때 호출됨 (방금 업데이트가 끝났다는 뜻)
        if self._last_update_step:
            print(f"[Update done  @ timesteps={self.model.num_timesteps}]")
            self._last_update_step = 0
