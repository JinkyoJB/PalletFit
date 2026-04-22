# planning/RL/SB3/param_tune/callbacks_param_tuning_v1.py
from __future__ import annotations
from pathlib import Path
import torch
from stable_baselines3.common.callbacks import BaseCallback

class SaveEvalCallback(BaseCallback):
    """
    ─ save_freq 마다 ───────────────────────────────────────────
      1) 모델 체크포인트 (*.zip) 저장
      2) 1-에피소드 평가 → 보상·배치 개수 콘솔 출력
      3) 정책 네트워크 weight 분포(Histogram) + 파라미터 L2-norm(TensorBoard)
    """
    def __init__(
        self,
        eval_env,
        save_dir: str | Path,
        save_freq: int = 500,
    ):
        super().__init__()
        self.eval_env  = eval_env
        self.save_freq = int(save_freq)
        self.save_dir  = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step % self.save_freq:          # save_freq 배수가 아니면 패스
            return True

        # 1) ── 모델 저장
        ckpt_path = self.save_dir / f"ckpt_{step:08d}"
        self.model.save(str(ckpt_path))

        # 2) ── 1-에피소드 평가
        obs, _    = self.eval_env.reset()
        ep_R      = 0.0
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = self.eval_env.step(action)
            ep_R += reward
            self.eval_env.bin.render(
                write_num=True,
                name=f"step{step:08d}_ep{self.eval_env._episode_idx:03d}",
                save=True,
                show=False,
                save_path=f"planning/RL/SB3/eval_snaps/ckpt_{step:08d}",
            )
            if term or trunc:
                break
        print(f"[Eval @ {step:8d}]  reward={ep_R:8.2f}  "
              f"placed={info['n_placed']:3d}  "
              f"fill={info['fill']:.3f}  imb={info['imbalance']:.2f}")

        # 3) ── TensorBoard: 네트워크 파라미터 히스토그램 & L2-norm
        writer = getattr(self.logger, "writer", None)
        if writer is not None and hasattr(writer, "add_histogram"):
            for name, p in self.model.policy.named_parameters():
                # TensorBoard는 numpy array 형식이 가장 호환성이 좋다
                writer.add_histogram(f"policy/{name}",
                                    p.data.cpu().numpy(),  # numpy로 변환
                                    global_step=step)
            # L2-norm
            total_norm = torch.norm(
                torch.stack([param.data.norm(2) for param in
                            self.model.policy.parameters()]))
            writer.add_scalar("policy/param_l2_norm", total_norm.item(), step)
            writer.flush()
        else:
            # tensorboardX가 너무 옛날 버전이거나 SummaryWriter가 없음
            print("[WARN] Histogram logger가 활성화되지 않았습니다.")


        return True
