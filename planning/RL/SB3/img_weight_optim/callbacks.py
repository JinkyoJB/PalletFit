from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json
import numpy as np
from PIL import Image

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

# ──────────────────────────────────────────────────────────────
# 공용: TensorBoard writer 핸들
def _get_tb_writer(logger) -> Optional["tensorboardX.SummaryWriter"]:
    if logger is None or not hasattr(logger, "output_formats"):
        return None
    for fmt in logger.output_formats:
        if isinstance(fmt, TensorBoardOutputFormat):
            return fmt.writer
    return None
# ──────────────────────────────────────────────────────────────


class SaveTopKCallback(BaseCallback):
    """
    일정 간격(save_freq)마다 1-에피소드 평가를 수행하고,
    모니터 지표(예: su/return) 기준으로 '상위 top_k개' 모델만 유지합니다.

    - monitor: "su" 또는 "return"
    - mode   : "max"(클수록 좋음) / "min"(작을수록 좋음)
    - top_k  : 보존할 상위 체크포인트 수
    - save_eval_snap: 평가 에피소드 마지막 프레임 PNG 저장 여부
    - write_param_hist: 파라미터 히스토그램/L2-norm을 TB에 기록
    - 상태는 save_dir/topk_index.json 에 보관되어 재시작 시 이어서 관리
    """
    def __init__(
        self,
        eval_env,
        save_dir: str | Path,
        save_freq: int = 2048,
        monitor: str = "su",              # "su" or "return"
        mode: str = "max",                   # "max" or "min"
        top_k: int = 3,
        save_eval_snap: bool = True,
        write_param_hist: bool = True,
        eval_n_episodes: int = 7,
        fixed_eval_indices: Optional[List[int]] = None,  # 예: [0,1,2,3,4,5,6]
        fixed_eval_paths: Optional[List[str]] = None,    # 직접 파일 경로 리스트
    ):
        super().__init__()
        self.eval_env         = eval_env
        self.save_freq        = int(save_freq)
        self.save_dir         = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        assert monitor in ("su", "return"), "monitor must be 'su' or 'return'"
        self.monitor          = monitor
        assert mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.mode             = mode
        self.top_k            = int(top_k)
        self.save_eval_snap   = bool(save_eval_snap)
        self.write_param_hist = bool(write_param_hist)

        # 인덱스 파일(재시작 지원)
        self.index_path       = self.save_dir / "topk_index.json"
        self.snaps_dir        = self.save_dir / "topk_snaps"
        self.snaps_dir.mkdir(parents=True, exist_ok=True)

        # 내부 상태: [{step:int, metric:float, filename:str}]
        self.top_list: List[Dict[str, Any]] = []
        self._load_index()

        self.eval_n_episodes   = int(eval_n_episodes)
        self.fixed_eval_indices = list(fixed_eval_indices) if fixed_eval_indices is not None else None
        self.fixed_eval_paths   = list(fixed_eval_paths) if fixed_eval_paths is not None else None


    # ───────── 도우미 ─────────
    def _eval_one(self) -> Tuple[float, dict, Optional[np.ndarray]]:
        """단일 에피소드 평가(기존 함수명을 보존)."""
        ep_R = 0.0
        last_info = {}
        final_img = None

        is_vec = hasattr(self.eval_env, "num_envs")
        if is_vec:
            obs = self.eval_env.reset()
            done = np.array([False] * self.eval_env.num_envs)
            while not np.any(done):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_R += float(rewards[0])   # n_envs=1 가정
                done = dones
                last_info = infos[0]
                if "final_image_hwc" in last_info:
                    final_img = last_info["final_image_hwc"]
        else:
            obs, _ = self.eval_env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_R += float(reward)
                last_info = info
                if (terminated or truncated) and "final_image_hwc" in last_info:
                    final_img = last_info["final_image_hwc"]
                if terminated or truncated:
                    break
        return ep_R, last_info, final_img

    def _evaluate_many_episodes(self) -> Tuple[float, float, Dict[str, Any], Optional[np.ndarray]]:
        """
        고정된 N개 에피소드를 순서대로 평가하여
        (return 평균, SU 평균, 참고 info, 마지막 스냅) 반환.
        """
        returns, sus, infos, snaps = [], [], [], []
        n = max(1, self.eval_n_episodes)

        for i in range(n):
            # ★ 다음 에피소드를 고정 지정 (env에 앞서 추가한 훅 사용)
            try:
                if hasattr(self.eval_env, "env_method"):  # VecEnv 계열
                    if self.fixed_eval_paths is not None:
                        p = self.fixed_eval_paths[i % len(self.fixed_eval_paths)]
                        self.eval_env.env_method("set_next_eval_episode", episode_path=p)
                    elif self.fixed_eval_indices is not None:
                        idx = int(self.fixed_eval_indices[i % len(self.fixed_eval_indices)])
                        self.eval_env.env_method("set_next_eval_episode", episode_index=idx)
                    else:
                        # 고정 0..6 같은 패턴
                        self.eval_env.env_method("set_next_eval_episode", episode_index=i)
            except Exception:
                # env가 해당 메서드를 제공하지 않으면 랜덤 선택(기존 동작)
                pass

            ep_R, info, final_img = self._eval_one()
            su = float(info.get("su", float("nan")))
            returns.append(ep_R)
            sus.append(su)
            infos.append(info)
            snaps.append(final_img)

        # 평균/통계
        ret_mean = float(np.nanmean(returns)) if len(returns) else float("nan")
        su_mean  = float(np.nanmean(sus))     if len(sus)     else float("nan")

        # 마지막 스냅(또는 SU가 가장 큰 에피소드 스냅을 쓰고 싶다면 아래 한 줄로 교체)
        final_snap = snaps[-1] if snaps else None
        # best_idx = int(np.nanargmax(sus)) if sus else -1; final_snap = snaps[best_idx] if best_idx >= 0 else None

        # 대표 info
        rep = dict(
            n_episodes=n,
            su_mean=su_mean,
            return_mean=ret_mean,
            su_list=[float(x) for x in sus],
            ret_list=[float(x) for x in returns],
            rep_episode=infos[-1] if infos else {},
        )
        return ret_mean, su_mean, rep, final_snap

    def _is_better(self, a: float, b: float) -> bool:
        return (a > b) if self.mode == "max" else (a < b)

    def _sort_key(self, item: Dict[str, Any]):
        # 정렬 키 (mode에 따라 오름/내림)
        return item["metric"] if self.mode == "min" else -item["metric"]

    def _load_index(self):
        if not self.index_path.is_file():
            return
        try:
            data = json.loads(self.index_path.read_text())
            items = data.get("items", [])
            kept = []
            for it in items:
                # 표준화해서 실제 파일이 있으면 keep
                p = self._ensure_zip(it["filename"])
                if p.exists():
                    kept.append(it)
            self.top_list = sorted(kept, key=self._sort_key)[: self.top_k]
            self._persist_index()
        except Exception:
            self.top_list = []

    def _persist_index(self):
        try:
            self.index_path.write_text(json.dumps({"items": self.top_list}, indent=2))
        except Exception:
            pass

    def _evaluate_one_episode(self) -> Tuple[float, dict, Optional[np.ndarray]]:
        ep_R = 0.0
        last_info = {}
        final_img = None

        is_vec = hasattr(self.eval_env, "num_envs")
        if is_vec:
            obs = self.eval_env.reset()
            done = np.array([False] * self.eval_env.num_envs)
            while not np.any(done):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_R += float(rewards[0])   # n_envs=1 가정
                done = dones
                last_info = infos[0]
                if "final_image_hwc" in last_info:
                    final_img = last_info["final_image_hwc"]
        else:
            obs, _ = self.eval_env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_R += float(reward)
                last_info = info
                if (terminated or truncated) and "final_image_hwc" in last_info:
                    final_img = last_info["final_image_hwc"]
                if terminated or truncated:
                    break

        return ep_R, last_info, final_img
    
    def _ensure_zip(self, name_wo_ext: str) -> Path:
        """
        'name', 'name.zip', 'name.zip.zip' 등 SB3 버전별로 저장된 실제 파일을 찾아
        최종적으로 'name.zip' 으로 맞춰서(필요시 rename) Path를 반환.
        """
        p_zip   = self.save_dir / f"{name_wo_ext}.zip"
        p_noext = self.save_dir / name_wo_ext
        p_dbl   = self.save_dir / f"{name_wo_ext}.zip.zip"

        if p_zip.exists():
            return p_zip
        if p_noext.exists():
            try:
                p_noext.rename(p_zip)  # 확장자 없으면 .zip 붙여 표준화
                return p_zip
            except Exception:
                return p_noext
        if p_dbl.exists():
            try:
                p_dbl.rename(p_zip)    # .zip.zip → .zip
                return p_zip
            except Exception:
                return p_dbl
        # 없으면 표준 경로 반환(상위 호출부에서 생성/무시)
        return p_zip

    def _save_model(self, step: int, metric: float) -> str:
        name = f"topk_{self.monitor}_step{step:08d}_metric{metric:.6f}"  # 확장자 없이
        target = self.save_dir / f"{name}.zip"
        self.model.save(str(target))  # 어떤 SB3라도 일단 저장 시도
        self._ensure_zip(name)        # 저장 결과를 'name.zip'으로 표준화
        return name

    def _save_snap(self, step: int, metric: float, final_img: Optional[np.ndarray]):
        if not self.save_eval_snap or final_img is None:
            return
        Image.fromarray(final_img).save(
            self.snaps_dir / f"topk_{self.monitor}_step{step:08d}_metric{metric:.6f}.png"
        )

    def _try_insert_topk(self, step: int, metric: float, filename_wo_ext: str) -> bool:
        """
        top_k 리스트에 (step, metric, filename) 삽입.
        - 삽입되면 True 반환
        - 삽입되지 않으면 False (즉, 저장했던 모델을 삭제해야 함)
        """
        entry = {"step": int(step), "metric": float(metric), "filename": filename_wo_ext}
        # 리스트가 k개 미만이면 무조건 삽입
        if len(self.top_list) < self.top_k:
            self.top_list.append(entry)
            self.top_list.sort(key=self._sort_key)
            self._persist_index()
            return True

        # 이미 k개면 최악과 비교 후 교체
        # 정렬되어 있으므로 최악은 끝/앞 쪽 중 하나
        worst = sorted(self.top_list, key=self._sort_key, reverse=False)[-1] if self.mode == "max" \
            else sorted(self.top_list, key=self._sort_key, reverse=True)[-1]

        if self._is_better(metric, worst["metric"]):
            # 새 항목 삽입
            self.top_list.append(entry)
            # 정렬 후 상위 k개만 유지
            self.top_list.sort(key=self._sort_key)
            to_keep = self.top_list[: self.top_k]
            to_del  = [it for it in self.top_list[self.top_k :]]
            self.top_list = to_keep

            # 디스크에서 삭제
            for it in to_del:
                # 표준화로 실제 파일 경로를 얻은 뒤 삭제
                try:
                    self._ensure_zip(it["filename"]).unlink()
                except Exception:
                    pass
                try:
                    (self.snaps_dir / f"{it['filename']}.png").unlink()
                except Exception:
                    pass

            self._persist_index()
            return True

        return False

    # ───────── SB3 훅 ─────────
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        step = self.num_timesteps
        if step % self.save_freq != 0:
            return

        # 1) N-에피소드 평가 (평균 SU/Return)
        ret_mean, su_mean, rep_info, final_img = self._evaluate_many_episodes()
        metric = su_mean if self.monitor == "su" else ret_mean

        # 2) 로깅
        self.logger.record("eval/return_mean", ret_mean)
        self.logger.record("eval/su_mean", su_mean)
        self.logger.record("eval/n_episodes", self.eval_n_episodes)
        # 참고로 직전 대표 에피소드 단일값도 남길 수 있음
        if "rep_episode" in rep_info:
            rep = rep_info["rep_episode"]
            if "n_placed" in rep: self.logger.record("eval/n_placed_rep", int(rep["n_placed"]))
            if "su" in rep:       self.logger.record("eval/su_rep", float(rep["su"]))
        if self.top_list:
            best_now = max(self.top_list, key=lambda x: x["metric"])["metric"] if self.mode == "max" \
                       else min(self.top_list, key=lambda x: x["metric"])["metric"]
            self.logger.record("eval/topk_best", float(best_now))
        self.logger.record("eval/monitored_metric", float(metric))
        self.logger.dump(step)

        print(f"[Eval @ {step:8d}] return_mean={ret_mean:7.2f} | su_mean={su_mean:6.3f} "
              f"| kept={len(self.top_list)}/{self.top_k}")

        # 3) 모델 저장 시도 (평균 metric 기준)
        fname = self._save_model(step, metric)  # (임시) 저장
        kept = self._try_insert_topk(step, metric, fname)

        if kept:
            print(f"[TopK] Kept: {fname}.zip")
            self._save_snap(step, metric, final_img)  # 스냅은 마지막(또는 best) 1장
        else:
            try:
                self._ensure_zip(fname).unlink()
            except Exception:
                pass

        # 4) 파라미터 히스토그램/L2 (기존 그대로)
        if self.write_param_hist:
            writer = _get_tb_writer(self.logger)
            if writer is not None:
                for name, p in self.model.policy.named_parameters():
                    writer.add_histogram(f"policy/{name}", p.data.cpu().numpy(), global_step=step)
                total_norm = torch.norm(torch.stack([param.data.norm(2) for param in self.model.policy.parameters()]))
                writer.add_scalar("policy/param_l2_norm", total_norm.item(), step)
                writer.flush()
                

class SaveEvalCallback(BaseCallback):
    """
    save_freq 마다:
      - eval 1 episode (deterministic)
      - eval 지표 로깅(self.logger.record + dump)
      - (옵션) 베스트 모델만 저장 (monitor, mode 기준)
      - (옵션) 마지막 모델 저장(save_last)
      - (옵션) info['final_image_hwc'] 스냅 저장
      - (옵션) 파라미터 히스토그램 + L2-norm (TB writer가 있을 때)
    """
    def __init__(
        self,
        eval_env,
        save_dir: str | Path,
        save_freq: int = 500,
        monitor: str = "su",          # "su" or "return"
        mode: str = "max",               # "max" or "min"
        save_best_only: bool = True,     # True면 최고 갱신시에만 저장
        save_last: bool = False,         # True면 매 eval마다 last_model.zip도 갱신
        save_eval_snap: bool = True,     # True면 마지막 프레임 PNG 저장
    ):
        super().__init__()
        self.eval_env       = eval_env
        self.save_freq      = int(save_freq)
        self.save_dir       = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor        = monitor
        assert self.monitor in ("su", "return"), "monitor must be 'su' or 'return'"
        self.mode           = mode
        assert self.mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.save_best_only = bool(save_best_only)
        self.save_last      = bool(save_last)
        self.save_eval_snap = bool(save_eval_snap)

        # best metric 초기화 + 복구
        self.best_metric_path = self.save_dir / "best_metric.json"
        if self.mode == "max":
            self.best_metric = -float("inf")
        else:
            self.best_metric = float("inf")
        self._restore_best_metric()

    # ---------- 내부 유틸 ----------
    def _restore_best_metric(self):
        try:
            if self.best_metric_path.is_file():
                data = json.loads(self.best_metric_path.read_text())
                val = float(data.get("best_metric", None))
                if val is not None:
                    self.best_metric = val
        except Exception:
            pass

    def _persist_best_metric(self):
        try:
            self.best_metric_path.write_text(json.dumps({"best_metric": self.best_metric}))
        except Exception:
            pass

    def _is_better(self, new: float, ref: float) -> bool:
        return (new > ref) if self.mode == "max" else (new < ref)

    def _evaluate_one_episode(self) -> Tuple[float, dict, Optional[np.ndarray]]:
        """eval_env에서 deterministic rollout 1회 수행."""
        ep_R = 0.0
        last_info = {}
        final_img = None

        is_vec = hasattr(self.eval_env, "num_envs")
        if is_vec:
            obs = self.eval_env.reset()
            done = np.array([False] * self.eval_env.num_envs)
            while not np.any(done):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                ep_R += float(rewards[0])  # n_envs=1 가정
                done = dones
                last_info = infos[0]
                if "final_image_hwc" in last_info:
                    final_img = last_info["final_image_hwc"]
        else:
            obs, _ = self.eval_env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_R += float(reward)
                last_info = info
                if (terminated or truncated) and "final_image_hwc" in last_info:
                    final_img = last_info["final_image_hwc"]
                if terminated or truncated:
                    break

        return ep_R, last_info, final_img

    # ---------- SB3 훅 ----------
    def _on_step(self) -> bool:
        # 매 스텝마다 아무 것도 안 함
        return True

    def _on_rollout_end(self) -> None:
        step = self.num_timesteps
        if step % self.save_freq != 0:
            return

        # 1) 평가
        ep_R, info, final_img = self._evaluate_one_episode()
        n_placed = int(info.get("n_placed", -1))
        su     = float(info.get("su", -1.0))

        # 2) 로깅
        self.logger.record("eval/return", ep_R)
        self.logger.record("eval/su", su)
        self.logger.record("eval/n_placed", n_placed)
        # best metric도 함께 기록
        self.logger.record("eval/best_metric", float(self.best_metric))
        self.logger.dump(step)

        print(f"[Eval @ {step:8d}] reward={ep_R:8.2f}  placed={n_placed:3d}  su={su:.3f}")

        # 3) 모니터링할 값 선택
        metric = su if self.monitor == "su" else ep_R

        # 4) last 모델 저장(옵션)
        if self.save_last:
            self.model.save(str(self.save_dir / "last_model"))

        # 5) best-only 저장
        if not self.save_best_only:
            # 원한다면 매번 체크포인트 (이전 방식)
            self.model.save(str(self.save_dir / f"ckpt_{step:08d}"))
        else:
            if self._is_better(metric, self.best_metric):
                self.best_metric = metric
                self._persist_best_metric()

                # 베스트 모델 저장 (고정 경로)
                best_path = self.save_dir / "best_model"
                self.model.save(str(best_path))

                # 스냅샷도 갱신
                if self.save_eval_snap and final_img is not None:
                    snap_dir = self.save_dir / "best_snaps"
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(final_img).save(snap_dir / "best_eval.png")

                print(f"[Eval @ {step:8d}] NEW BEST {self.monitor}={metric:.4f} → saved: {best_path.with_suffix('.zip').name}")

        # 6) (옵션) 파라미터 히스토그램 + L2-norm (TB writer가 있을 때)
        writer = _get_tb_writer(self.logger)
        if writer is not None:
            # 히스토그램
            for name, p in self.model.policy.named_parameters():
                writer.add_histogram(f"policy/{name}", p.data.cpu().numpy(), global_step=step)
            # L2-norm
            total_norm = torch.norm(
                torch.stack([param.data.norm(2) for param in self.model.policy.parameters()])
            )
            writer.add_scalar("policy/param_l2_norm", total_norm.item(), step)
            writer.flush()