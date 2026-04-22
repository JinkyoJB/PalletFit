'''
# 예시 1: 최종 모델로 3개 에피소드, 결정론적으로 실행
python planning/RL/SB3/heu_rewards_RL/test.py \
  --model $PALLETFIT_ROOT/planning/RL/SB3/heu_rewards_RL/logs_20250910/final_model.zip \
  --episode_dir planning/data/Item_data/paper/setting123_discrete \
  --n_episodes 3 --det

# 예시 2: best 모델로 3개 에피소드, 스토캐스틱 실행
python planning/RL/SB3/heu_rewards_RL/test.py \
  --model $PALLETFIT_ROOT/planning/RL/SB3/heu_rewards_RL/logs_20250910/best/best_model.zip \
  --n_episodes 3 --stochastic

# 예시 3: 특정 에피소드 파일만 고정해서 순환 테스트
python planning/RL/SB3/heu_rewards_RL/test.py \
  --model $PALLETFIT_ROOT/planning/RL/SB3/heu_rewards_RL/logs_20250910/final_model.zip \
  --fixed_eps planning/data/Item_data/paper/setting123_discrete/dataset_episode_619.json \
              planning/data/Item_data/paper/setting123_discrete/dataset_episode_1001.json

# 예시 4: 최종 모델로 3개 에피소드
python planning/RL/SB3/heu_rewards_RL/test.py \
  --model $PALLETFIT_ROOT/planning/RL/SB3/heu_rewards_RL/logs/ckpt/auxppo_20000_steps.zip\
  --episode_dir planning/data/Item_data/paper/setting123_discrete \
  --n_episodes 3 --det
'''
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

import torch as th
import gymnasium as gym

# ▶ 학습 코드에서 정의한 커스텀 알고리즘/정책/래퍼를 그대로 import
from planning.RL.SB3.heu_rewards_RL.train import (
    EpsMaskablePPO,
    ValidCheck,
)
from planning.RL.SB3.heu_rewards_RL.env import ImgHeurObsRewardEnv
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


# ============== 유틸 ==============
def find_vecnorm_path(model_path: str) -> str | None:
    """모델 경로 근처에서 vecnorm.pkl 탐색"""
    p = Path(model_path).resolve()
    candidates = []
    if p.is_file():
        candidates += [p.parent / "vecnorm.pkl", p.parent.parent / "vecnorm.pkl"]
    else:
        candidates += [p / "vecnorm.pkl", p / "logs" / "vecnorm.pkl"]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None

def make_eval_vecenv_with_norm(base_env: gym.Env, vecnorm_path: str | None):
    """단일 gym.Env -> DummyVecEnv -> (옵션) VecNormalize 로 감싸기"""
    # Monitor로 감싸면 SB3가 자동으로 또 감싸지 않아 깔끔
    env = Monitor(base_env)
    venv = DummyVecEnv([lambda: env])
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        # 최소한 관측 스케일만 맞춰주고 학습모드는 끄기
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
    return venv

def to_torch_obs_vec(obs, device):
    """VecEnv용: 이미 (B=1, …) 배치가 있으므로 unsqueeze 금지"""
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                out[k] = th.tensor(v.copy(), dtype=th.float32, device=device)
            else:
                out[k] = v
        return out
    return th.tensor(np.array(obs, copy=True), dtype=th.float32, device=device)

def hwc_from_chw_u8(arr: np.ndarray) -> np.ndarray:
    """
    (3, H, W) uint8 → (H, W, 3) uint8
    이미 (H, W, 3)라면 그대로 반환
    """
    if arr.ndim == 3 and arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))
    return arr


def to_jsonable(x):
    """np.ndarray/np.generic 등을 JSON 가능 형태로 재귀 변환"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    # 최후의 보루: repr로 직렬화
    try:
        return repr(x)
    except Exception:
        return "<unserializable>"


def save_obs_triplet(obs: dict, out_dir: Path, basename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, tag in [("image_top", "top"), ("image_left", "left"), ("image_front", "front")]:
        if key not in obs:
            continue
        img = hwc_from_chw_u8(obs[key])
        Image.fromarray(img).save(out_dir / f"{basename}_{tag}.png")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============== Env 빌더 (평가용) ==============

def build_eval_env(
    episode_dir: str,
    *,
    seed: int | None = 123,
    use_valid_check: bool = True,
):
    base = ImgHeurObsRewardEnv(
        render_every=0,                # bin.render()는 테스트 루프에서 직접 호출
        item_seed=seed,
        item_mode="episode_dir",
        episode_dir=episode_dir,
        is_eval_env=True,              # ▶ set_next_eval_episode() 사용 가능
        worker_id=900,
        debug_img_dir=None,            # (중복 저장 방지: 직접 저장할 것이라 0)
        debug_save_every=0,
        log_to_console=True,
        log_dir=None,
    )

    # 액션 마스킹
    def _mask_fn(_obs):
        return base.unwrapped.valid_action_mask()

    env: gym.Env = ActionMasker(base, _mask_fn)
    if use_valid_check:
        env = ValidCheck(env)
    return env


# ============== 액션 추론(마스크 반영) ==============

def to_torch_obs_with_batch(obs, device):
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                t = th.tensor(v.copy(), dtype=th.float32, device=device)
                if t.dim() >= 1:
                    t = t.unsqueeze(0)  # B=1
                out[k] = t
            else:
                out[k] = v
        return out
    t = th.tensor(np.array(obs, copy=True), dtype=th.float32, device=device)
    if t.dim() >= 1:
        t = t.unsqueeze(0)
    return t


# ============== 모델 경로 해석 ==============

def resolve_model_path(model_path: str) -> str:
    """입력으로 파일 또는 디렉토리를 받아, 실제 로드 가능한 .zip 경로를 찾아 반환.
    우선순위: best/best_model.zip > final_model.zip > ckpt/ 최신 zip > 디렉토리 내 최신 zip
    """
    p = Path(model_path)

    # 1) 명시된 경로가 실제 zip 파일이면 그대로 사용
    if p.is_file():
        return str(p)

    # 2) 디렉토리라면 내부를 스캔
    if p.is_dir():
        candidates = [p / "best" / "best_model.zip", p / "final_model.zip"]
        for c in candidates:
            if c.is_file():
                return str(c)
        # ckpt 디렉토리 최신 zip
        ckpt_dir = p / "ckpt"
        if ckpt_dir.is_dir():
            zips = sorted(ckpt_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
            if zips:
                return str(zips[0])
        # 디렉토리 전체 재귀 검색(최신)
        zips = sorted(p.rglob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
        if zips:
            return str(zips[0])
        raise FileNotFoundError(f"No .zip model found under directory: {p}")

    # 3) 파일로 보이지만 존재하지 않을 때: 부모 폴더에서 추론
    parent = p.parent
    if parent.exists():
        # /logs/final_model.zip 가 없을 때 /logs/best/best_model.zip, /logs/ckpt 최신 zip 시도
        for guess in [parent / "best" / "best_model.zip", parent / "final_model.zip"]:
            if guess.is_file():
                return str(guess)
        ckpt_dir = parent / "ckpt"
        if ckpt_dir.is_dir():
            zips = sorted(ckpt_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
            if zips:
                return str(zips[0])
    # 마지막 시도: 상위 두 단계까지 스캔
    for up in [p.parent, p.parent.parent]:
        if up and up.exists() and up.is_dir():
            for guess in [up / "best" / "best_model.zip", up / "final_model.zip"]:
                if guess.is_file():
                    return str(guess)
            ckpt_dir = up / "ckpt"
            if ckpt_dir.is_dir():
                zips = sorted(ckpt_dir.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
                if zips:
                    return str(zips[0])

    raise FileNotFoundError(f"Model not found: {model_path}. Try passing a directory like .../logs or a full path to best_model.zip")


# ============== 메인 테스트 루프 ==============

def run_eval(
    model_path: str,
    episode_dir: str,
    out_dir: str,
    n_episodes: int,
    deterministic: bool,
    render_bin_every: int,
    fixed_episode_paths: list[str] | None = None,
    seed: int | None = 123,
    use_valid_check: bool = True,
):
    out_root = ensure_dir(Path(out_dir))

    # 1) 단일 base env 만들기 (기존 그대로)
    base_env = build_eval_env(episode_dir, seed=seed, use_valid_check=use_valid_check)

    # 2) 모델 경로/vecnorm 경로 결정
    resolved_model_path = resolve_model_path(model_path)
    vecnorm_path = find_vecnorm_path(resolved_model_path)
    print(f"[Test] Using model: {resolved_model_path}")
    print(f"[Test] VecNormalize stats: {vecnorm_path or 'NONE (will use eval-only norm)'}")

    # 3) VecEnv + VecNormalize 로 감싸기 (학습과 동일 스케일)
    env = make_eval_vecenv_with_norm(base_env, vecnorm_path)

    # 4) 커스텀 클래스 로드 (env=VecEnv!)
    model: EpsMaskablePPO = EpsMaskablePPO.load(resolved_model_path, env=env, device="auto")
    device = model.device

    # (고정 에피소드 지정용으로 base_env 접근이 필요하므로 helper)
    def _base_unwrapped():
        e = env
        # VecNormalize -> DummyVecEnv -> Monitor -> ValidCheck -> ActionMasker -> Base
        if isinstance(e, VecNormalize): e = e.venv
        if hasattr(e, "envs"): e = e.envs[0]
        if hasattr(e, "env"): e = e.env
        return e.unwrapped

    episode_paths = [str(Path(p)) for p in fixed_episode_paths] if fixed_episode_paths else None


    # 실행 로그 파일
    run_meta = {
        "model_path": str(model_path),
        "episode_dir": str(episode_dir),
        "n_episodes": int(n_episodes),
        "deterministic": bool(deterministic),
        "render_bin_every": int(render_bin_every),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "episode_paths": episode_paths or [],
    }
    with open(out_root / "run_meta.json", "w", encoding="utf-8") as fp:
        json.dump(run_meta, fp, ensure_ascii=False, indent=2)

    for ep in range(n_episodes):
        if episode_paths:
            _base_unwrapped().set_next_eval_episode(episode_path=episode_paths[ep % len(episode_paths)])

        # ✅ VecEnv는 배치가 이미 포함됨
        obs = env.reset()                    # dict of np.ndarray, each (1, …)
        done = False
        ep_dir = ensure_dir(out_root / f"ep_{ep:03d}")
        obs_dir = ensure_dir(ep_dir / "obs")
        bin_dir = ensure_dir(ep_dir / "bin")

        meta_path = ep_dir / "episode_log.jsonl"
        steps = 0
        ep_reward = 0.0
        with open(meta_path, "w", encoding="utf-8") as mfp:
            while not done:
                # ── 마스크: vec 방식으로 꺼내기
                mask = env.env_method("valid_action_mask")[0]         # 1개 env
                mask_b = np.asarray(mask, dtype=bool)[None, :]        # (1, n_actions)

                # ── 배치 obs 그대로 토치로
                obs_t = to_torch_obs_vec(obs, device)

                with th.no_grad():
                    dist = model.policy.get_distribution(obs_t, action_masks=mask_b)
                    act_t = dist.get_actions(deterministic=deterministic)

                # ── VecEnv step: 액션도 배치형으로
                action_b = act_t.detach().cpu().numpy()               # shape (1,)
                obs, reward_b, done_b, info_b = env.step(action_b)

                # 로깅/집계
                r = float(reward_b[0]); d = bool(done_b[0])
                ep_reward += r; done = d

                # ── 원본 관측/렌더 저장은 base env에서
                base = f"st_{steps:04d}"
                # VecNormalize 전에 base obs를 저장하려면 _base_unwrapped().last_obs 같은 내부값이 없다면
                # 여기서는 정규화된 obs를 역변환하지 않고, 저장은 옵션으로 두세요.
                # 필요 시 VecNormalize의 normalize_obs/unnormalize_obs를 활용.

                if render_bin_every > 0 and (steps % render_bin_every == 0):
                    try:
                        b = _base_unwrapped()
                        b.bin.render(save=True, show=False, save_path=str(bin_dir),
                                     write_num=True, name=f"{base}_SU{b.bin.SU:.3f}")
                    except Exception as e:
                        print(f"[warn] bin.render failed at step {steps}: {e}")

                rec = {"step": steps, "action": int(action_b[0]), "reward": r, "done": d}
                mfp.write(json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n")

                steps += 1

        # 에피소드 마지막 렌더
        try:
            b = _base_unwrapped()
            b.bin.render(save=True, show=False, save_path=str(bin_dir),
                         write_num=True, name=f"st_{steps:04d}_final_SU{b.bin.SU:.3f}")
        except Exception:
            pass

        with open(ep_dir / "summary.json", "w", encoding="utf-8") as fp:
            json.dump({"ep_reward": ep_reward, "steps": steps}, fp, ensure_ascii=False, indent=2)
        print(f"[Test] Episode {ep} done | steps={steps} | ep_reward={ep_reward:.3f}")

    env.close()
    print("[Test] All episodes finished. Outputs saved to:", out_root)

# ============== CLI ==============

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained (Aux)MaskablePPO and save stepwise images.")
    parser.add_argument("--model", dest="model_path", type=str,
                        default="planning/RL/SB3/heu_rewards_RL/logs",
                        help="모델 경로: .zip 파일 또는 logs 디렉토리(자동 탐색)")
    parser.add_argument("--episode_dir", type=str,
                        default="planning/data/Item_data/paper/setting123_discrete",
                        help="테스트에 사용할 episode json 디렉토리")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="출력 루트 디렉토리. 지정 안하면 logs/test_run/<timestamp>")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--det", dest="deterministic", action="store_true", help="deterministic 액션 사용")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="stochastic 액션 사용")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--render_bin_every", type=int, default=1, help="bin.render 저장 주기(스텝)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no_valid_check", action="store_true", help="유효 액션 강제 가드 비활성화")
    parser.add_argument("--fixed_eps", nargs="*", default=None,
                        help="고정 에피소드 파일 경로 리스트 (공백 구분). 지정하면 해당 파일들만 순환 테스트")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"planning/RL/SB3/heu_rewards_RL/test_run/{timestamp}"

    run_eval(
        model_path=args.model_path,
        episode_dir=args.episode_dir,
        out_dir=out_dir,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render_bin_every=args.render_bin_every,
        fixed_episode_paths=args.fixed_eps,
        seed=args.seed,
        use_valid_check=not args.no_valid_check,
    )


if __name__ == "__main__":
    main()
