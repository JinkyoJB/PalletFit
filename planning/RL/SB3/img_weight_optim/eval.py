# planning/RL/SB3/img_weight_optim/eval.py
from __future__ import annotations
import argparse, time
from statistics import mean
from pathlib import Path
import numpy as np
from sb3_contrib import RecurrentPPO
from planning.RL.SB3.img_weight_optim.env import ImgWeightOptimEnv

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = "planning/RL/SB3/img_weight_optim/checkpoints/250903_final.zip", help="path to .zip model (will try to add .zip if missing)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--device", default="cpu")
    return p.parse_args()

def resolve_model_path(p: str | Path) -> Path:
    p = Path(p)
    if p.suffix != ".zip":
        p_zip = p.with_suffix(".zip")  # 확장자 없이 준 경우 대비
        if p_zip.exists():
            return p_zip
    return p

if __name__ == "__main__":
    args = parse()
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 평가 환경 (훈련 때와 동일 설정 권장)
    env = ImgWeightOptimEnv(
        render_every=0,
        log_dir="planning/RL/SB3/img_weight_optim/logs/eval_run",
        # 필요하면 item_mode/episode_dir 등을 훈련과 동일하게 설정
    )

    # ★ 모델 로드: 첫 인자는 경로
    model = RecurrentPPO.load(
        model_path,
        device=args.device,
        print_system_info=True,
    )

    R_all, placed_all, su_all = [], [], []  # SU(= fill %) 추가
    t0 = time.time()

    state = None  # LSTM state
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        episode_start = True
        R = 0.0
        last_info = {}

        while True:
            t_infer = time.time()
            action, state = model.predict(
                obs,
                state=state,
                episode_start=np.array([episode_start], dtype=bool),
                deterministic=True,
            )
            episode_start = False
            # print(f"[DEBUG] action: {action} (inference {time.time()-t_infer:.3f}s)")

            t_step = time.time()
            obs, r, term, trunc, info = env.step(action)
            R += float(r)
            last_info = info
            print(f"[DEBUG] step: {env._step} (env.step {time.time()-t_step:.3f}s)")

            # 원하면 스냅 저장
            t_render = time.time()
            env.bin.render(write_num=True, name=f"ep{ep:03d}_step{env._step:04d}",
                           save=True, show=False,
                           save_path=f"planning/RL/SB3/img_weight_optim/eval_snaps/{model_path.stem}")
            # print(f"[DEBUG] render: {env._step} (render {time.time()-t_render:.3f}s)")

            if term or trunc:
                break

        # 에피소드별 기록
        R_all.append(R)
        placed_all.append(last_info.get("n_placed", 0))
        su_all.append(last_info.get("su", float("nan")))  # ← fill% (SU) 수집

        print(f"[EP {ep:03d}] R={R:6.2f} | placed={last_info.get('n_placed', -1)} | SU={last_info.get('su', float('nan')):.3f}%")

        # 다음 에피소드용 LSTM state 초기화
        state = None

    # 요약
    print("\n── Summary ──")
    print(f"Reward  : {mean(R_all):.2f} ± {np.std(R_all):.2f}")
    print(f"Placed  : {mean(placed_all):.2f} ± {np.std(placed_all):.2f}")
    # SU
    print(f"SU(fill%): {np.nanmean(su_all):.3f}% ± {np.nanstd(su_all):.3f}%")
    print(f"Elapsed : {time.time()-t0:.1f}s")

    env.close()
