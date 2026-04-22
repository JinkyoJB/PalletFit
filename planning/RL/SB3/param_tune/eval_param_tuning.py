# planning/RL/SB3/param_tune/eval_param_tuning.py
from __future__ import annotations
import argparse, time
from statistics import mean
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from planning.RL.SB3.param_tune.env_param_tuning_v1 import ParamTuneEnv

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help=".zip 모델")
    p.add_argument("--episodes", type=int, default=1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse()
    env = ParamTuneEnv(render_every=0,
                       log_dir="planning/RL/SB3/param_tune/logs/eval_run")
    model = PPO.load(args.model, env=env, device="cpu")

    R_all, placed_all = [], []
    t0 = time.time()
    for ep in range(1, args.episodes+1):
        obs, _ = env.reset(); R = 0.0
        while True:
            act,_ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act); R += r
            env.bin.render(
                write_num=True, 
                name=f"ep{ep:03d}_step{env._step:04d}",
                save=True, 
                show=False, 
                save_path=f"planning/RL/SB3/eval_snaps/{args.model.split('/')[-1].replace('.zip', '')}" 
            )
            if term or trunc: break
        R_all.append(R); placed_all.append(info["n_placed"])
        print(f"[EP {ep:03d}] R={R:6.2f} | placed={info['n_placed']}")

    print("\n── Summary ──")
    print(f"Reward  : {mean(R_all):.2f} ± {np.std(R_all):.2f}")
    print(f"Placed  : {mean(placed_all):.2f} ± {np.std(placed_all):.2f}")
    print(f"Elapsed : {time.time()-t0:.1f}s")
    env.close()
