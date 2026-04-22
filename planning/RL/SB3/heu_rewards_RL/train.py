# planning/RL/SB3/heu_rewards_RL/train.py
import os, torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)             # 각 env 프로세스 내 BLAS 오버구독 방지
torch.backends.cudnn.benchmark = True  # 입력 크기 고정이면 CNN 속도↑

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from pathlib import Path

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList,ProgressBarCallback, BaseCallback
)
from planning.RL.SB3.heu_rewards_RL.env import ImgHeurObsRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# e-psilon 스케줄러 (학습 초반에 많이 탐색)
def headstart_linear(start: float = 0.3, head_frac: float = 0.3):
    """
    학습 앞부분(head_frac: 0~1)에서만 start→0으로 선형 감소, 이후 0 유지.
    progress_remaining: 1.0(시작) → 0.0(종료)
    """
    def fn(progress_remaining: float) -> float:
        # 앞 head_frac 구간에만 유효
        x = (progress_remaining - (1 - head_frac)) / head_frac
        x = max(0.0, min(1.0, x))
        return start * x
    return fn

# ============== 유효액션 강제 가드 ==============
class ValidCheck(gym.Wrapper):
    def step(self, action):
        assert self.env.unwrapped.valid_action_mask()[int(action)], f"Invalid action picked: {action}"
        return super().step(action)

# ===== ε-탐욕 전용 MaskablePPO =====
class EpsMaskablePPO(MaskablePPO):
    def __init__(self, *args, eps_explore=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # float 또는 callable(progress_remaining)->float 둘 다 허용
        self.eps_explore = eps_explore

    def _vec_get_action_masks(self, env, use_masking: bool):
        if not use_masking:
            return None
        if hasattr(env, "env_method"):
            masks_list = env.env_method("valid_action_mask")
            return np.stack([np.asarray(m, dtype=bool) for m in masks_list], axis=0)
        if hasattr(env, "valid_action_mask"):
            m = env.valid_action_mask()
            return np.asarray(m, dtype=bool)[None, :]
        return None

    @staticmethod
    def _to_torch_obs(obs, device):
        if isinstance(obs, dict):
            return {
                k: th.as_tensor(v, dtype=th.float32, device=device) if isinstance(v, np.ndarray) else v
                for k, v in obs.items()
            }
        return th.as_tensor(obs, dtype=th.float32, device=device)

    # sb3-contrib 2.x 시그니처
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int, use_masking: bool):
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()

        if self._last_obs is None:
            reset_out = env.reset()
            self._last_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            self._last_episode_starts = np.ones((env.num_envs,), dtype=bool)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            action_masks = self._vec_get_action_masks(env, use_masking)
            obs_for_policy = self._to_torch_obs(self._last_obs, self.device)

            with th.no_grad():
                dist = self.policy.get_distribution(obs_for_policy, action_masks=action_masks)
                actions_t = dist.get_actions(deterministic=False)

                # ----- ε-탐욕 (Top-M valid 내 랜덤; M은 필요시 튜닝) -----
                eps = float(self.eps_explore(self._current_progress_remaining)) \
                    if callable(self.eps_explore) else float(self.eps_explore)
                TOP_M = 64

                if eps > 0.0:
                    actions = actions_t.detach().cpu().numpy()
                    n_envs = env.num_envs
                    n_actions = self.action_space.n
                    am = np.ones((n_envs, n_actions), dtype=bool) if action_masks is None else action_masks.astype(bool)

                    # 분포 확률 (마스크 적용 전)
                    if hasattr(dist.distribution, "probs"):
                        probs = dist.distribution.probs
                    else:
                        probs = th.softmax(dist.distribution.logits, dim=-1)
                    probs_np = probs.detach().cpu().numpy()

                    for i in range(n_envs):
                        if np.random.rand() < eps:
                            valid = np.flatnonzero(am[i])
                            if valid.size > 0:
                                pv = probs_np[i, valid]
                                m = min(TOP_M, valid.size)
                                top_m_idx = np.argpartition(-pv, m - 1)[:m]
                                top_m = valid[top_m_idx]
                                actions[i] = np.random.choice(top_m)
                    actions_t = th.as_tensor(actions, device=self.device, dtype=th.long)

                log_probs = dist.log_prob(actions_t)
                values = self.policy.predict_values(obs_for_policy)

                if hasattr(dist.distribution, "probs"):
                    probs = dist.distribution.probs
                else:
                    probs = th.softmax(dist.distribution.logits, dim=-1)
                mode_actions = probs.argmax(dim=-1)
                is_explore = (actions_t != mode_actions)
                ent = dist.entropy()

            actions = actions_t.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            expl_np = is_explore.detach().cpu().numpy()
            ent_np = ent.detach().cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            # 디버그 메타데이터
            for i in range(env.num_envs):
                if not isinstance(infos[i], dict):
                    infos[i] = {}
                a = int(actions[i])
                pa = float(probs_np[i, a])
                pmax = float(probs_np[i].max())
                rank = int((-probs_np[i]).argsort().tolist().index(a))
                infos[i].update({
                    "a_idx": a,
                    "a_prob": pa,
                    "a_top_prob": pmax,
                    "a_rank": rank,
                    "a_entropy": float(ent_np[i]),
                    "a_is_explore": bool(expl_np[i]),
                })

            n_steps += 1
            self.num_timesteps += env.num_envs

            if len(self.ep_info_buffer) > 0:
                ep_rew_mean = np.mean([e["r"] for e in self.ep_info_buffer])
                ep_len_mean = np.mean([e["l"] for e in self.ep_info_buffer])
                self.logger.record("rollout/ep_rew_mean", ep_rew_mean)
                self.logger.record("rollout/ep_len_mean", ep_len_mean)
                self.logger.record("explore/eps", eps)

            rollout_buffer.add(
                obs=self._last_obs,
                action=actions,
                reward=rewards,
                episode_start=self._last_episode_starts,
                value=values,
                log_prob=log_probs,
                action_masks=action_masks,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones.astype(bool)

            callback.update_locals(locals())
            if not callback.on_step():
                return False

        with th.no_grad():
            values = self.policy.predict_values(self._to_torch_obs(self._last_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)

        callback.on_rollout_end()
        return True

    def train(self):
        self.policy.set_training_mode(True)

        # 스케줄 해석
        clip_range = float(self.clip_range(self._current_progress_remaining)) \
            if callable(self.clip_range) else float(self.clip_range)
        if self.clip_range_vf is None:
            clip_range_vf = None
        else:
            clip_range_vf = float(self.clip_range_vf(self._current_progress_remaining)) \
                if callable(self.clip_range_vf) else float(self.clip_range_vf)

        # ★ 엔트로피 계수 스케줄도 숫자로 평가
        ent_coef = float(self.ent_coef(self._current_progress_remaining)) \
            if callable(self.ent_coef) else float(self.ent_coef)

        # lr 스케줄 반영
        self._update_learning_rate(self.policy.optimizer)

        ent_losses, pg_losses, vf_losses, losses = [], [], [], []
        approx_kl_list, clip_frac_list, ev_list = [], [], []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = actions.long().flatten()

                # (옵션) 마스크 row 비어있을 때 안전장치
                am = rollout_data.action_masks
                if am is not None:
                    row_empty = (am.sum(dim=-1) == 0)
                    if row_empty.any():
                        am[row_empty, 0] = True

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, action_masks=rollout_data.action_masks
                )

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                values_pred = values.flatten()
                if clip_range_vf is None:
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                else:
                    values_pred_clipped = rollout_data.old_values + th.clamp(
                        values_pred - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    value_losses = (values_pred - rollout_data.returns) ** 2
                    value_losses_clipped = (values_pred_clipped - rollout_data.returns) ** 2
                    value_loss = 0.5 * th.mean(th.max(value_losses, value_losses_clipped))

                entropy_loss = -th.mean(entropy)

                # ★ 여기서 ent_coef는 float
                loss = policy_loss + self.vf_coef * value_loss + ent_coef * entropy_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # 로깅
                pg_losses.append(policy_loss.item())
                vf_losses.append(value_loss.item())
                ent_losses.append(entropy_loss.item())
                losses.append(loss.item())

                with th.no_grad():
                    approx_kl = (rollout_data.old_log_prob - log_prob).mean().item()
                    clip_frac = (th.abs(ratio - 1.0) > clip_range).float().mean().item()
                    ev = explained_variance(
                        rollout_data.returns.detach().cpu().numpy(),
                        values_pred.detach().cpu().numpy()
                    )
                approx_kl_list.append(approx_kl)
                clip_frac_list.append(clip_frac)
                ev_list.append(ev)

            self._n_updates += 1
            self.logger.record("train/n_updates", self._n_updates)
            self.logger.record("train/policy_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(vf_losses))
            self.logger.record("train/entropy_loss", np.mean(ent_losses))
            self.logger.record("train/loss", np.mean(losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_list))
            self.logger.record("train/clip_fraction", np.mean(clip_frac_list))
            self.logger.record("train/explained_variance", np.mean(ev_list))
            self.logger.record("train/clip_range", clip_range)
            if clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)
            self.logger.record("train/ent_coef", ent_coef)  # ★ 숫자 로깅


class FixedEpisodeEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        episode_paths: list[str] | None = None,
        episode_indices: list[int] | None = None,
        eval_freq: int = 10_000,
        n_episodes: int = 7,
        deterministic: bool = True,
        best_model_save_path: str | None = None,
        tb_prefix: str = "fixed_eval",
        verbose: int = 0,
        # ▶ 추가: 이미지 저장 옵션
        save_eval_images: bool = False,
        save_dir: str = "planning/RL/SB3/heu_rewards_RL/eval_img",
    ):
        super().__init__(verbose)
        assert (episode_paths is not None) or (episode_indices is not None)
        self.eval_env = eval_env
        self.episode_paths = episode_paths
        self.episode_indices = episode_indices
        self.eval_freq = int(eval_freq)
        self.n_episodes = int(n_episodes)
        self.deterministic = bool(deterministic)
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = None
        self.tb_prefix = tb_prefix
        # ▶ 추가 필드
        self.save_eval_images = bool(save_eval_images)
        self.save_dir = str(save_dir)

    @staticmethod
    def _to_torch_obs_with_batch(obs, device):
        """
        Dict[np.ndarray] -> Dict[torch.FloatTensor] 로 캐스팅하면서
        배치 차원(B=1)을 앞에 붙여준다.
        """
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    # 읽기 전용 경고 회피: 복사본으로 텐서 생성
                    t = th.tensor(v.copy(), dtype=th.float32, device=device)
                    # 벡터/행렬/이미지/3D 텐서 모두 B 차원 추가
                    if t.dim() >= 1:
                        t = t.unsqueeze(0)  # (1, …)
                    out[k] = t
                else:
                    out[k] = v
            return out
        # dict가 아닌 경우도 동일 처리
        t = th.tensor(np.array(obs, copy=True), dtype=th.float32, device=device)
        if t.dim() >= 1:
            t = t.unsqueeze(0)
        return t

    def _enable_img_saving(self, enable: bool):
        base = self.eval_env.unwrapped  # 래퍼 아래 실제 Env
        if enable:
            p = Path(self.save_dir); p.mkdir(parents=True, exist_ok=True)
            base.debug_dir = p
            base.debug_save_every = 1     # 매 스텝 저장
        else:
            base.debug_save_every = 0     # 저장 끔

    def _set_episode(self, j: int):
        if self.episode_paths is not None:
            p = self.episode_paths[j % len(self.episode_paths)]
            self.eval_env.unwrapped.set_next_eval_episode(episode_path=p)
        else:
            idx = self.episode_indices[j % len(self.episode_indices)]
            self.eval_env.unwrapped.set_next_eval_episode(episode_index=idx)

    @staticmethod
    def _to_torch_obs(obs, device):
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    out[k] = th.tensor(v.copy(), dtype=th.float32, device=device)
                else:
                    out[k] = v
            return out
        return th.tensor(np.array(obs, copy=True), dtype=th.float32, device=device)


    def _base_env(self):
        # VecNormalize -> DummyVecEnv -> Monitor -> ValidCheck -> ActionMasker -> Base
        e = self.eval_env
        while hasattr(e, "envs"):       # VecEnv 래퍼 벗기기
            e = e.envs[0]
        return e.unwrapped

    def _set_episode(self, j: int):
        base = self._base_env()
        if self.episode_paths is not None:
            p = self.episode_paths[j % len(self.episode_paths)]
            base.set_next_eval_episode(episode_path=p)
        else:
            idx = self.episode_indices[j % len(self.episode_indices)]
            base.set_next_eval_episode(episode_index=idx)

    def _enable_img_saving(self, enable: bool):
        base = self._base_env()
        if enable:
            p = Path(self.save_dir); p.mkdir(parents=True, exist_ok=True)
            base.debug_dir = p; base.debug_save_every = 1
        else:
            base.debug_save_every = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        rewards = []
        device = self.model.device

        for j in range(self.n_episodes):
            self._enable_img_saving(self.save_eval_images and j == 0)
            self._set_episode(j)

            # ✅ VecEnv는 배치가 이미 포함됨: reset() 그대로 받기
            obs = self.eval_env.reset()      # dict of np.ndarray, each with shape (1, …)
            done = False
            ep_rew = 0.0

            while not done:
                # ✅ 마스크: env_method는 리스트를 돌려주니 [0]만 꺼내고 배치 차원(1) 붙여줌
                mask = self.eval_env.env_method("valid_action_mask")[0]
                mask_b = np.asarray(mask, dtype=bool)[None, :]  # (1, n_actions)

                # ✅ 배치가 이미 있으므로 unsqueeze 금지: _to_torch_obs 사용
                obs_t = self._to_torch_obs(obs, device)

                with th.no_grad():
                    dist = self.model.policy.get_distribution(obs_t, action_masks=mask_b)
                    act_t = dist.get_actions(deterministic=self.deterministic)

                # ✅ VecEnv는 액션도 배치형으로
                action = act_t.detach().cpu().numpy()  # shape: (1,)

                # ✅ VecEnv step 시그니처: (obs, rewards, dones, infos)
                obs, rewards_b, dones_b, infos = self.eval_env.step(action)

                ep_rew += float(rewards_b[0])
                done = bool(dones_b[0])

            rewards.append(ep_rew)

        if self.save_eval_images:
            self._enable_img_saving(False)

        mean_rew = float(np.mean(rewards))
        self.model.logger.record(f"{self.tb_prefix}/mean_reward", mean_rew)
        if self.verbose:
            print(f"[FixedEval] mean_reward({self.n_episodes} eps) = {mean_rew:.3f}")

        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
            if (self.best_mean_reward is None) or (mean_rew > self.best_mean_reward):
                self.best_mean_reward = mean_rew
                self.model.save(os.path.join(self.best_model_save_path, "best_model.zip"))
        return True



# ============== 학습용 환경 빌더 (단일 env) ==============
def build_train_env(episode_dir: str, seed: int = 42, log_dir: str | None = None):
    base = ImgHeurObsRewardEnv(
        render_every=0, item_seed=seed, item_mode="episode_dir",
        episode_dir=episode_dir, is_eval_env=False, worker_id=0,
        debug_img_dir=None, debug_save_every=0, log_to_console=True,
        log_dir=log_dir,
    )
    def mask_fn(_obs):
        return base.unwrapped.valid_action_mask()
    env = ActionMasker(base, mask_fn)
    env = ValidCheck(env)
    return env

# ============== 학습용 환경 빌더 (병렬 env) ==============
def make_env_thunk(episode_dir, log_root, seed, rank):
    def _thunk():
        env = ImgHeurObsRewardEnv(
            render_every=0, item_seed=seed + rank, item_mode="episode_dir",
            episode_dir=episode_dir, is_eval_env=False, worker_id=rank,
            debug_img_dir=None, debug_save_every=0, log_to_console=True,
            log_dir=os.path.join(log_root, f"train_env_{rank}")  # 각 워커별 로그 분리
        )
        # 마스크는 각 프로세스 안에서 만들어 주면 pickle 이슈 없음
        def mask_fn(_obs):
            return env.unwrapped.valid_action_mask()
        env = ActionMasker(env, mask_fn)
        env = ValidCheck(env)
        return env
    return _thunk

def make_eval_vecenv(Episode_DIR: str):
    def _thunk():
        base = ImgHeurObsRewardEnv(
            render_every=0, item_seed=123, item_mode="episode_dir",
            episode_dir=Episode_DIR, is_eval_env=True, worker_id=99,
            debug_img_dir=None, debug_save_every=0, log_to_console=False, log_dir=None
        )
        env = ActionMasker(base, lambda _obs: base.unwrapped.valid_action_mask())
        env = ValidCheck(env)
        env = Monitor(env)
        return env
    venv = DummyVecEnv([_thunk])                     # ✅ VecEnv로 감싸기
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
    return venv

# ============== 실행부 ==============
def main():
    EPISODE_DIR = "planning/data/Item_data/paper/setting123_discrete"
    LOG_DIR = "planning/RL/SB3/heu_rewards_RL/logs"
    SAVE_DIR = f"{LOG_DIR}/ckpt"
    BEST_DIR = f"{LOG_DIR}/best"

    EPISODE_DIR = "planning/data/Item_data/paper/setting123_discrete"
    FIXED_EPISODES = [
        "dataset_episode_619.json",
        "dataset_episode_1323.json",
        "dataset_episode_1001.json",
        "dataset_episode_1778.json",
        "dataset_episode_818.json",
        "dataset_episode_2249.json",
        "dataset_episode_621.json",
    ]
    EVAL_EP_PATHS = [str(Path(EPISODE_DIR) / name) for name in FIXED_EPISODES]

    # 학습용 env (병렬)
    N_ENVS = 16  # 원하는 병렬 개수
    N_STEPS  = 1024     # per-env
    ROLLOUT  = N_ENVS * N_STEPS 
    BATCH    = 2048    # ROLLOUT의 약수로 잡기
    # N_ENVS = 1  # 원하는 병렬 개수
    # N_STEPS  = 16     # per-env
    # ROLLOUT  = N_ENVS * N_STEPS 
    # BATCH    = 16    # ROLLOUT의 약수로 잡기
    env = SubprocVecEnv([make_env_thunk(EPISODE_DIR, LOG_DIR, seed=42, rank=i) for i in range(N_ENVS)])
    env = VecMonitor(env)  # ep_rew_mean, ep_len_mean 로깅
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # 단일 env (디버그용)
    # env = build_train_env(EPISODE_DIR, seed=42, log_dir=LOG_DIR)
    # 평가용 별도 env (eval=True로 만들면 set_next_eval_episode가 활성 동작)
    eval_env = make_eval_vecenv(EPISODE_DIR)

    # 학습용 VecNormalize 통계 복사
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = getattr(env, "ret_rms", None)
    eval_env.training = False
    eval_env.norm_reward = False
    
    ent_schedule = get_linear_fn(0.02, 0.0, 1.0)
    eps_schedule = get_linear_fn(0.20, 0.00, 0.30)

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=128),

        net_arch=[256, 256],
    )

    model = EpsMaskablePPO(
        policy=MaskableActorCriticPolicy,
        policy_kwargs=policy_kwargs,
        env=env,
        n_steps=N_STEPS,
        batch_size=BATCH,               # 4096 (rollout=16384이면 OK)
        n_epochs=5,                     # 10 → 5 (조금 더 빈번히 샘플 교체)
        learning_rate=1e-3,            # 기본 3e-4 → 1e-3로 과감히 ↑ (효과 없으면 5e-4~2e-3 탐색)
        clip_range=0.3,                # 0.2 → 0.3 (clip_fraction 5~20% 목표)
        vf_coef=1.0,                   # 0.5 → 1.0 (가치함수 안정화)
        max_grad_norm=0.8,             # 0.5~1.0 권장
        ent_coef=ent_schedule,
        eps_explore=eps_schedule,
        tensorboard_log=f"{LOG_DIR}/tb",
        verbose=1,
    )

    # ❶ 주기 저장: 20,000 스텝마다 auxppo_*.zip 저장
    ckpt_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=SAVE_DIR,
        name_prefix="auxppo",
        save_replay_buffer=False,  # PPO는 리플레이 버퍼 없음
    )

    # ❷ 베스트 모델 저장(선택): 1,000 스텝마다 평가하여 최고 성능 저장
    # 고정 7개 에피소드 평가 콜백
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

    callback = CallbackList([
        progress_cb,
        ckpt_cb,
        fixed_eval_cb,
    ])

    model.learn(total_timesteps=100_000, callback=callback)

    env.save(f"{LOG_DIR}/vecnorm.pkl")
    model.save(f"{LOG_DIR}/final_model.zip")

if __name__ == "__main__":
    main()
