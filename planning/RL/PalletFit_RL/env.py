# planning/RL/PalletFit_RL/env.py
from __future__ import annotations
from dataclasses import dataclass, is_dataclass
from typing import Optional, Dict, Any, Deque, List
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from planning.data.data_sources import DataSource, make_source
from planning.RL.PalletFit_RL.config import (
    PREVIEW_MAX, ACTION_MAX_CANDIDATES, PIVOT_FEAT_DIM,
    OBS_TOPK_DEFAULT, ITEM_FEAT_DIM, GLOBAL_FEAT_DIM,
)

from planning.RL.PalletFit_RL.reward_builder import build_reward, get_failure_penalty, get_terminal_bonus
from planning.RL.PalletFit_RL.obs_builder import build_obs, make_obs_space_gym, queue_head
from planning.RL.PalletFit_RL.act_builder import place_by_action, rebuild_candidates

from planning.item import RotationType, Item
from planning.bin import Bin
from planning.BinSpecsDict import BIN_SPECS
from planning.packer import Packer

# ─────────────────────────────────────────────────────────────
# (옛 _generate_synthetic_items 헬퍼는 SyntheticSource로 흡수됨 — 2026-05-04 DataSource 추상화)
# ─────────────────────────────────────────────────────────────
# Gym Env
# ─────────────────────────────────────────────────────────────


class PalletFitEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        *,
        seed: int = 0,
        tb_log_dir: Optional[str | Path] = None,
        max_retry_per_step: int = 20,
        is_render_env: bool = False,
        gif_fps: int = 4,
    ):
        super().__init__()
        self._seed = int(seed)
        self._tb_log_dir = Path(tb_log_dir) if tb_log_dir else None
        self._is_render_env = bool(is_render_env)
        self._gif_fps = int(gif_fps)

        # [수정] Config의 PREVIEW_MAX를 사용하여 차원 결정
        self._N_max = int(PREVIEW_MAX)
        self._K = int(ACTION_MAX_CANDIDATES)
        self._TOTAL = self._N_max * self._K
        self.NOOP_IDX = self._TOTAL - 1
        
        # ---------------- Observation / Action spaces ----------------
        base_obs_space = make_obs_space_gym()
        self.observation_space = spaces.Dict({
            **base_obs_space.spaces,
            "act_mask":  spaces.Box(0.0, 1.0, shape=(self._TOTAL,), dtype=np.float32),
            "act_cands": spaces.Box(0.0, 1.0, shape=(self._TOTAL, PIVOT_FEAT_DIM), dtype=np.float32),
        })
        self.max_steps_per_episode = 100
        
        # 통합된 변수 초기화 (기본값: Config상수)
        self._preview_cnt = int(PREVIEW_MAX)

        self.action_space = spaces.Discrete(self._TOTAL, start=0)

        # 내부 상태
        self._pending_plan: Optional[Dict[str, Any]] = None
        self._plan_in_use: Optional[Dict[str, Any]] = None

        self._episode_idx = 0
        self._steps_in_ep = 0
        self._ep_return = 0.0

        self._ep_reward_terms = defaultdict(float)   # 에피소드 누적

        self._last_reset_meta: Optional[Dict[str, Any]] = None
        self._head_indices: List[int] = []

        self.packer = None
        self.queue: Deque[int] = deque()

        # 액션 마스크 / 직전 obs 캐시
        self._last_cands: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

        # GIF 렌더 (eval 0번 워커 전용)
        self._gif_capture_active: bool = False
        self._gif_save_dir: Optional[Path] = None
        self._gif_frames: List[np.ndarray] = []
        self._gif_episode_idx: int = -1

        # 시드 적용
        random.seed(self._seed)
        np.random.seed(self._seed)

        self.max_retry_per_step = max_retry_per_step
        self._current_step_retry_count = 0

        # ── reward delta용 baseline 점수 ───────────────────────
        # build_reward가 state-based 절대 점수를 반환하므로,
        # env가 step n과 n+1의 점수 차이를 reward로 사용한다.
        # placement-specific 항목(alive/stab/contact)은 일회성 보너스이므로
        # _finalize_step에서 prev_score 갱신 시 빼낸다.
        self._prev_score: float = 0.0
        # term별 delta 계산용 baseline (state-only). reset/_finalize_step에서 갱신.
        self._prev_terms: Dict[str, float] = {}
        # 어떤 term이 placement-specific(다음 step baseline에서 빼야 함)인지
        self._PLACEMENT_TERMS = ("alive", "stab_soft", "qual_contact")

    # 추가: 관측/마스크를 NOOP-only로 만드는 안전 상태
    def _make_noop_only_state(self):
        self._last_mask = np.zeros((self._TOTAL,), dtype=bool)
        self._last_mask[self.NOOP_IDX] = True
        self._last_cands = np.zeros((self._TOTAL, PIVOT_FEAT_DIM), dtype=np.float32)

    def _make_zero_obs(self) -> Dict[str, np.ndarray]:
        """observation_space 형상에 맞는 0-채움 obs. invalid 에피소드 등 rebuild/obs를 돌릴 가치가 없는 경로용."""
        return {
            "items_topk":    np.zeros((OBS_TOPK_DEFAULT, ITEM_FEAT_DIM), dtype=np.float32),
            "items_mask":    np.zeros((OBS_TOPK_DEFAULT,), dtype=np.float32),
            "globals":       np.zeros((GLOBAL_FEAT_DIM,), dtype=np.float32),
            "preview_queue": np.zeros((PREVIEW_MAX, 4), dtype=np.float32),
            "act_mask":      self._last_mask.astype(np.float32),
            "act_cands":     self._last_cands.astype(np.float32),
        }

    # ── VecEnv/콜백에서 쓰는 헬퍼 ─────────────────────
    def get_SU(self) -> float:
        return float(self._bin.SU) if self._bin is not None else 0.0

    def pop_last_reset_meta(self):
        meta = self._last_reset_meta
        self._last_reset_meta = None
        return meta
    
    def apply_plan(self, plan: Any):
        """
        외부(agent)에서 플랜을 지정.
        - plan: dict 또는 EnvPlan (dataclass)
        - 다음 reset() 시점에 _pending_plan이 적용된다.
        """
        plan_dict = self._as_plan_dict(plan)
        self._pending_plan = plan_dict
        self._plan_in_use = plan_dict           # 현재 에피소드 info에 노출할 '사용 중 플랜'도 같이 갱신

    def _as_plan_dict(self, plan: Any) -> Dict[str, Any]:
        """dict 또는 EnvPlan → 통일된 dict 형식으로 변환.

        새 형식 (2026-05-04, DataSource 추상화):
            {
                "source":      {"mode": "...", "args": {...}},   # ← 박스 source 명세
                "bin":         "experiment_RL",                  # ← bin alias (None이면 source.bin_alias_hint)
                "bin_payload": {margin_x, margin_y, preview_cnt, max_steps_per_episode},
                "tag":         "train" | "eval_*",                # logging/GIF 캡처
                "seed":        episode seed,
            }
        """
        default = {
            "source": {
                "mode": "recorded",
                "args": {"paths": ["planning/data/Item_data/exhibition/real_object_0331.json"]},
            },
            "bin": "experiment_RL",
            "bin_payload": {},
            "tag": "train",
            "seed": self._seed,
        }

        if plan is None:
            return dict(default)

        # dataclass → dict
        if is_dataclass(plan):
            from dataclasses import asdict
            plan = asdict(plan)

        if not isinstance(plan, dict):
            return dict(default)

        out = dict(default)
        if "source" in plan:
            out["source"] = dict(plan["source"])
        out["bin"] = str(plan.get("bin", plan.get("bin_alias", out["bin"])))
        out["tag"] = str(plan.get("tag", plan.get("mode", out["tag"])))
        out["seed"] = int(plan.get("seed", out["seed"]))

        # bin_payload merge (top-level alias 흡수)
        bin_payload = dict(plan.get("bin_payload", {}) or {})
        for src, dst in (("bin_margin_x", "margin_x"),
                         ("bin_margin_y", "margin_y"),
                         ("preview_cnt", "preview_cnt"),
                         ("max_steps_per_episode", "max_steps_per_episode")):
            if src in plan and dst not in bin_payload:
                bin_payload[dst] = plan[src]
        out["bin_payload"] = bin_payload
        return out


    # ── info 빌더 ────────────────────────────────────
    
    
    def _build_info(self, *, terminal_reason: Optional[str] = None) -> Dict[str, Any]:
        # source 정보 (DataSource 추상화 — info에 mode/id 노출, agent의 history가 이걸로 record)
        src = getattr(self, "_current_source", None)
        source_mode = src.mode if src is not None else "unknown"
        source_id   = src.source_id() if src is not None else "unknown"
        ep_name = f"{source_mode}_ep{self._episode_idx}"

        preview_cnt = int(self._preview_cnt)
        SU_now = float(self.get_SU())
        packed_count = int(self._bin.size) if self._bin else 0

        info = {
            "episode_path": ep_name,
            "source_mode": source_mode,
            "source_id":   source_id,
            "return": float(self._ep_return),
            "SU": SU_now,
            "packed_count": packed_count,
            "terminal_reason": terminal_reason or "",
            "plan": self._plan_in_use,
            "steps_in_ep": int(self._steps_in_ep),
            "preview_cnt": preview_cnt,
        }
        if getattr(self, "_ep_reward_terms", None):
            for k, v in self._ep_reward_terms.items():
                info[f"r_ep_{k}"] = float(v)
        return info

    # ── MaskablePPO용 액션 마스크 ────────────────────
    def action_masks(self) -> np.ndarray:   # 외부 호출용
        m = self._last_mask
        if m is None or m.size == 0 or not bool(m.any()):
            # 모두 불가능하면 최소한 NOOP 허용
            m = np.zeros((self._TOTAL,), dtype=bool)
            m[self.NOOP_IDX] = True
            self._last_mask = m
        return self._last_mask
    
    def _capture_frame(self) -> None:
        """현재 bin을 (H,W,3) uint8 ndarray로 캡처해 _gif_frames에 누적."""
        if not self._gif_capture_active or self._bin is None:
            return
        try:
            arr = self._bin.render(
                save=False,
                show=False,
                return_array=True,
                size_annotation=False,
                write_num=False,
            )
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                self._gif_frames.append(arr)
        except Exception as e:
            print(f"[GIF] capture failed at ep{self._episode_idx} step{self._steps_in_ep}: {e}")

    def _flush_gif(self) -> None:
        """누적 프레임을 GIF로 저장하고 버퍼 리셋."""
        if not self._gif_frames or self._gif_save_dir is None:
            self._gif_frames = []
            return
        try:
            import imageio.v2 as imageio
            self._gif_save_dir.mkdir(parents=True, exist_ok=True)
            su_now = float(self._bin.SU) if self._bin is not None else 0.0
            name = f"ep{self._gif_episode_idx:04d}_SU{su_now:.3f}.gif"
            out_path = self._gif_save_dir / name
            duration = max(1.0 / float(self._gif_fps), 1e-3)
            imageio.mimsave(str(out_path), self._gif_frames, duration=duration, loop=0)
            print(f"[GIF] saved {out_path} ({len(self._gif_frames)} frames)")
        except Exception as e:
            print(f"[GIF] save failed: {e}")
        finally:
            self._gif_frames = []

    def _safe_rebuild_candidates(self, check: bool = False):
        try:
            # rebuild_candidates 호출 인자 간소화 (preview_cnt 전달)
            self._last_mask, self._last_cands, self._head_indices = rebuild_candidates(
                TOTAL=self._TOTAL,
                preview_cnt=self._preview_cnt,
                queue=self.queue,
                bin=self._bin,
                K=self._K,
                NOOP_IDX=self.NOOP_IDX,
                cands_option='EDP',
                check=check
            )
        except Exception as e:
            print(f"[ERROR] _rebuild_candidates crashed: {e}")
            self._make_noop_only_state()
                
    # ── Gym API: reset/step ──────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        - self._pending_plan 이 있으면 그것을 사용하고,
        - 없으면 기존 self._plan_in_use 를 재사용,
        - 그것도 없으면 기본 플랜(offline, 빈 페이로드)을 사용한다.
        """
        self._episode_idx += 1
        self._steps_in_ep = 0
        self._ep_return = 0.0

        self._ep_reward_terms = defaultdict(float)
        if seed is not None:
            self._seed = int(seed)
            random.seed(self._seed)
            np.random.seed(self._seed)

        # ── 1) 이번 에피소드에 사용할 plan 확정 ─────────────
        # GIF 캡처는 "이번 reset이 새 pending_plan을 소비"할 때만 활성화 →
        # SB3 자동 재리셋(같은 plan 반복) 시 중복 GIF 방지.
        consumed_pending = self._pending_plan is not None
        if self._pending_plan is not None:
            plan = self._pending_plan
            self._pending_plan = None
            self._plan_in_use = plan
        elif self._plan_in_use is not None:
            plan = self._plan_in_use
        else:
            # 완전 초기 상태면 기본 플랜 생성
            plan = self._as_plan_dict(None)
            self._plan_in_use = plan

        # 평가 태그(eval_*) + 0번 워커 + 새 plan 소비일 때만 GIF 캡처
        tag = str(plan.get("tag", ""))
        if (
            self._is_render_env
            and consumed_pending
            and tag.startswith("eval")
            and self._tb_log_dir is not None
        ):
            self._gif_capture_active = True
            self._gif_save_dir = self._tb_log_dir / "eval_renders"
            self._gif_save_dir.mkdir(parents=True, exist_ok=True)
            self._gif_frames = []
            self._gif_episode_idx = int(self._episode_idx)
        else:
            self._gif_capture_active = False
            self._gif_frames = []

        # ── 2) bin 구성 ───────────────────────────────────────
        self.packer = self._build_binPacker_from_plan(plan)
        if self.packer is None:
            self.packer = Packer(rotation_type=RotationType.BasicRotation, order_setting=False)
            self.packer.build_bin(plan.get("bin", "experiment_RL"))

        # ── 3) 아이템 리스트 빌드 ─────────────────────────
        ok_items = self._build_items_from_plan(plan)
        if not ok_items:
            # 아이템 빌드 실패: rebuild_candidates/build_obs 모두 스킵.
            # NOOP-only 마스크 + 0-채움 obs면 충분하다 (다음 step에서 NOOP→즉시 종료).
            self._make_noop_only_state()
            self._last_obs = self._make_zero_obs()
            self._prev_score = 0.0
            info = self._build_info(terminal_reason="invalid_item_build_failed")
            info["invalid_episode"] = True
            self._last_reset_meta = {
                "episode_idx": int(self._episode_idx),
                "episode_path": info["episode_path"],
            }
            return self._last_obs, info

        # ── 6) 정상 빌드 완료
        self._safe_rebuild_candidates(check=True)

        head_q = queue_head(self._preview_cnt, self.queue)
        obs = build_obs(head_q, self._bin, self._last_mask, self._last_cands, self._TOTAL, self._preview_cnt)
        self._last_obs = obs

        # reward delta용 baseline 초기화 (state-only score & terms)
        try:
            initial_score, initial_terms = build_reward(self._bin, placed_item=None)
            self._prev_score = float(initial_score)
            self._prev_terms = {k: float(v) for k, v in initial_terms.items()}
        except Exception:
            self._prev_score = 0.0
            self._prev_terms = {}

        info = self._build_info()
        self._last_reset_meta = {
            "episode_idx": int(self._episode_idx),
            "episode_path": info["episode_path"],
        }
        # 빈 bin 첫 프레임
        self._capture_frame()
        return obs, info

    def step(self, action: int):
        # ─────────────────────────────────────────────────────────
        # 1. 초기 체크 (타임아웃, 아이템 없음 등)
        # ─────────────────────────────────────────────────────────
        # [리팩토링] before_bin = copy.deepcopy(self._bin) 제거.
        # build_reward가 state-based로 바뀌어 _prev_score(스칼라)로 delta를 계산.

        self._steps_in_ep += 1

        # 타임아웃
        if self._steps_in_ep >= self.max_steps_per_episode:
            return self._finalize_step(
                terminated=False, truncated=True, terminal_reason="max_steps_reached"
            )
        # 아이템 오링
        if len(self.queue) == 0:
            return self._finalize_step(terminated=True, truncated=False, terminal_reason="no_items")

        # ─────────────────────────────────────────────────────────
        # 2. 액션 실행 시도
        # ─────────────────────────────────────────────────────────
        if action == self.NOOP_IDX:
            # NOOP은 "포기" 선언이므로 종료 처리
            return self._finalize_step(
                terminated=True, truncated=False, terminal_reason="no_op_selected", failure_code="NOOP"
            )

        # 액션 시도
        ok, result = place_by_action(
            action,
            self._preview_cnt,
            self.queue, self._bin,
            self._last_mask, self._last_cands, self._K
        )

        if ok:
            # ✅ 성공: 정상적으로 적재하고 다음 스텝으로 진행
            self._current_step_retry_count = 0  # [리셋]

            placed_item = result
            self.packer.bins[self.packer.current_bin_idx] = self._bin
            if placed_item is not None:
                try:
                    self.queue.remove(placed_item._id)
                except ValueError:
                    pass
            # self._bin.simplify()

            return self._finalize_step(
                terminated=False, truncated=False, terminal_reason=None,
                placed_item=placed_item,
            )

        else:
            # ❌ 실패 (물리적 충돌 등): 죽이지 않고 "다시 해봐" 기회 제공
            error_code = result     # 예: -3 (FAIL_COLLISION)
            
            # (1) 마스크 끄기
            self._last_mask[action] = False
            self._last_obs["act_mask"] = self._last_mask.astype(np.float32)
            
            # (2) 재시도 카운트 증가
            self._current_step_retry_count += 1
            
            # (3) 벌점 부여 (치명적이지 않게 -0.5 ~ -1.0 정도)
            # reward_builder를 호출하되, terminated=False로 호출하여 벌점만 가져옴
            step_penalty = -0.01  # 재시도 벌점 (너무 크면 아무것도 안 하려고 하니 작게)

            # (4) 종료 조건: 더 이상 갈 곳이 없거나 OR ★재시도 횟수 초과★
            all_masked = not np.any(self._last_mask[:self.NOOP_IDX])
            too_many_retries = self._current_step_retry_count >= self.max_retry_per_step

            if all_masked or too_many_retries:
                # 종료 시 리셋
                self._current_step_retry_count = 0

                reason = "all_actions_failed" if all_masked else "max_retries_exceeded"

                return self._finalize_step(
                    terminated=True, truncated=False,
                    terminal_reason=reason,
                    failure_code="RETRY_LIMIT"  # 실패 코드 전달
                )

            # (5) 재시도 상태 반환 (terminated=False)
            # info에 실패 사유를 적어주면 디버깅에 좋음
            info = self._build_info()
            info["retry_reason"] = f"Action {action} failed ({error_code})"
            
            return self._last_obs, step_penalty, False, False, info

    # ─────────────────────────────────────────────────────────
    # [헬퍼] 종료 처리 함수
    # ─────────────────────────────────────────────────────────
    def _finalize_step(
        self,
        *,
        terminated: bool,
        truncated: bool,
        terminal_reason: str | None,
        placed_item=None,
        failure_code=None,
    ):
        # finished: "정상 진행/완료" 케이스. 그 외 terminated 사례(NOOP, RETRY_LIMIT 등)는 실패.
        finished = bool(terminal_reason in [
            "no_items",          # 모든 아이템 적재 완료
            "all_items_placed",  # (호환용) 동일 의미
            "max_steps_reached", # truncate
        ])

        if terminated and not finished:
            # 실패 종료 → 페널티만 부여. terms는 이미 per-step 기여도이므로 그대로 누적.
            reward, terms = get_failure_penalty(failure_code)
            delta_terms: Dict[str, float] = {k: float(v) for k, v in terms.items()}
            # _prev_terms / _prev_score는 갱신하지 않음 (bin state 변화 없음)
        else:
            # 정상 step / finished 종료 → state 점수의 delta로 reward 계산.
            curr_score, terms = build_reward(self._bin, placed_item=placed_item)
            reward = float(curr_score - self._prev_score)

            # ── term별 delta = (curr - prev) 로 분해 ─────────────────
            # state-based(eff_su/eff_dead/bal): prev에 동일 키가 있어 차이만큼만 기여.
            # placement-specific(alive/stab/contact): prev에는 0(또는 미존재)이므로
            # delta == 그 step의 전액 → reward 분해와 정확히 일치.
            delta_terms = {
                k: float(v) - float(self._prev_terms.get(k, 0.0))
                for k, v in terms.items()
            }

            # 다음 step의 baseline은 "state-only" 점수여야 한다.
            # placement-specific bonus는 일회성이므로 다음 step prev에서 빼낸다.
            placement_only = sum(
                terms.get(k, 0.0) for k in self._PLACEMENT_TERMS
            )
            self._prev_score = float(curr_score - placement_only)
            # _prev_terms도 동일 원칙: state-only로 스냅샷 (placement-specific은 0으로)
            self._prev_terms = {
                k: (0.0 if k in self._PLACEMENT_TERMS else float(v))
                for k, v in terms.items()
            }

            # ★ 옵션 D: finished(정상 종료/truncate) 시 terminal SU 보너스 일괄 지급.
            #   per-step ΔSU 분배를 없앤 대신 여기서 final_SU에 비례한 큰 한 방을 줘서
            #   진짜 목표(최종 SU)와 정책 학습 신호를 정렬.
            if (terminated or truncated) and finished:
                t_bonus, t_terms = get_terminal_bonus(self._bin)
                reward += float(t_bonus)
                for k, v in t_terms.items():
                    delta_terms[k] = delta_terms.get(k, 0.0) + float(v)

        self._ep_return += reward
        for k, v in delta_terms.items():
            self._ep_reward_terms[k] += float(v)

        # terminated일 땐 SB3가 다음 obs를 쓰지 않으므로(value bootstrap은 truncated에서만)
        # rebuild/obs 비용을 모두 스킵. 직전 _last_obs를 그대로 반환해 형상만 유지.
        if not terminated:
            self._safe_rebuild_candidates(check=True)
            head_q = queue_head(self._preview_cnt, self.queue)
            self._last_obs = build_obs(
                head_q, self._bin, self._last_mask, self._last_cands,
                self._TOTAL, self._preview_cnt,
            )
        info = self._build_info(terminal_reason=terminal_reason or None)

        # 성공적인 placement면 프레임 누적, 종료면 GIF 저장
        if placed_item is not None:
            self._capture_frame()
        if (terminated or truncated) and self._gif_capture_active:
            self._flush_gif()
            self._gif_capture_active = False

        return self._last_obs, float(reward), bool(terminated), bool(truncated), info
    
    # ── 내부 유틸 ─────────────────────────────────────

    def _build_items_from_plan(self, plan) -> bool:
        """plan["source"] spec → DataSource → sample(seed) 한 줄로 박스 생성.

        DataSource 추상화 (2026-05-04, docs/4 Sec 2-2 해결):
        이전 3-way 분기(synthetic/recorded/type_sampled)가 하나의 호출로 통합됨.
        새 source 추가 시 data_sources.py에 클래스 1개 등록만 하면 됨.
        """
        if self.packer is None:
            return False

        spec = plan.get("source") or {}
        seed = int(plan.get("seed", self._seed))
        self._seed = seed

        # 안전망 (docs/4 Sec 2-6 해결, 2026-05-05): synthetic source는 plan["bin"]을
        # 자동 주입해 bin_size를 BIN_SPECS와 동기화. 사용자가 spec에 bin_size/bin_alias를
        # 명시했으면 그게 우선 (override).
        if spec.get("mode") == "synthetic":
            args = dict(spec.get("args", {}))
            if "bin_size" not in args and "bin_alias" not in args:
                bin_alias = plan.get("bin")
                if bin_alias:
                    args["bin_alias"] = bin_alias
                    spec = {**spec, "args": args}

        try:
            # 단일 호출: spec → DataSource → 박스 dict 리스트
            self._current_source = make_source(spec)
            item_dicts = self._current_source.sample(seed)
            if not item_dicts:
                self.packer.items_list = []
                self.queue = deque()
                return False

            # dict → Item 객체로 변환 (필요한 default 채움)
            items: list[Item] = []
            for d in item_dicts:
                d = dict(d)
                d.setdefault("priority", 7)
                d.setdefault("updown", False)
                d.setdefault("options", {"color": "#14ba5e"})
                d.setdefault("weight", 0)
                d.setdefault("loadbear", 0)
                d.setdefault("unit", "mm")
                # rotation_quat이 list가 아니면 기본 회전으로 보정
                if not isinstance(d.get("rotation_quat"), list):
                    d["rotation_quat"] = list(RotationType.RT_WHD)
                items.append(Item(**d))

            self.packer.items_list = items
            self.queue = deque([it._id for it in items])
            return len(self.queue) > 0

        except Exception as e:
            print(f"[WARN] _build_items_from_plan failed: {e}")
            self.packer.items_list = []
            self.queue = deque()
            self._current_source = None
            return False


    def _build_binPacker_from_plan(self, plan) -> Packer:
        packer = Packer(rotation_type=RotationType.BasicRotation, order_setting=False)

        # bin alias 우선순위: plan["bin"] > source.bin_alias_hint() > "experiment_RL"
        bin_alias = plan.get("bin")
        if not bin_alias:
            spec = plan.get("source")
            if spec:
                try:
                    hint = make_source(spec).bin_alias_hint()
                except Exception:
                    hint = None
                bin_alias = hint
        bin_alias = bin_alias or "experiment_RL"

        bin_payload = plan.get("bin_payload", {}) or {}
        self.max_steps_per_episode = int(bin_payload.get("max_steps_per_episode", 100))
        
        self._preview_cnt = int(bin_payload.get("preview_cnt", PREVIEW_MAX))

        if bin_alias in BIN_SPECS:
            packer.build_bin(bin_alias)
            packer.current_bin.margin_x = int(bin_payload.get("margin_x", 0))
            packer.current_bin.margin_y = int(bin_payload.get("margin_y", 0))
        else:
            print(f"[WARN] Unknown bin_alias '{bin_alias}'. Fallback to 'experiment_RL'.")
            packer.build_bin("experiment_RL")

        return packer

    @property
    def _bin(self) -> Optional[Bin]:
        if self.packer is None:
            return None
        return self.packer.current_bin

    @property
    def _items_list(self) -> List[Item]:
        if self.packer is None:
            return []
        return self.packer.items_list

    def render(self):
        return None

    def close(self):
        return super().close()


# ─────────────────────────────────────────────────────────────────────
# 디버그 진입점: 랜덤(마스크 적용) 액션으로 env 돌려보기
#   사용 예) python -m planning.RL.PalletFit_RL.env --episodes 3 --gif
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import time
    import traceback

    parser = argparse.ArgumentParser(description="PalletFitEnv random-action debug runner")
    parser.add_argument("--episodes", type=int, default=2, help="에피소드 수")
    parser.add_argument("--max-steps", type=int, default=100, help="에피소드당 step 상한")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-cnt", type=int, default=3)
    parser.add_argument("--mode", choices=["synthetic", "recorded", "type_sampled"], default="synthetic",
                        help="아이템 소스")
    parser.add_argument("--recorded-path", type=str, default=None,
                        help="--mode recorded일 때 사용할 json 경로")
    parser.add_argument("--type-sampled-path", type=str, default=None,
                        help="--mode type_sampled일 때 사용할 json 경로")
    parser.add_argument("--gif", action="store_true",
                        help="에피소드별 GIF 저장 (eval tag로 0번 워커 처럼 동작)")
    parser.add_argument("--out-dir", type=str, default="planning/RL/PalletFit_RL/_debug_run",
                        help="GIF/로그 저장 위치")
    parser.add_argument("--bin-alias", type=str, default="experiment_RL")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # GIF가 켜져 있으면 0번-워커 흉내: is_render_env=True + tag=eval_*
    env = PalletFitEnv(
        seed=args.seed,
        tb_log_dir=args.out_dir if args.gif else None,
        is_render_env=bool(args.gif),
        gif_fps=4,
    )

    def _make_plan(ep_idx: int) -> Dict[str, Any]:
        ep_seed = int(args.seed + ep_idx * 1000)
        tag_prefix = "eval" if args.gif else "debug"
        # DataSource spec 만들기 (data_sources.spec_* 헬퍼와 동일 형식)
        if args.mode == "synthetic":
            spec = {
                "mode": "synthetic",
                "args": {
                    "bin_size": (1000, 1000, 1000),
                    "max_items": 30,
                    "min_item_mm": 100,
                    "max_aspect_ratio": 3.0,
                },
            }
        elif args.mode == "recorded":
            paths = [args.recorded_path] if args.recorded_path else []
            spec = {"mode": "recorded", "args": {"paths": paths}}
        else:  # type_sampled
            paths = [args.type_sampled_path] if args.type_sampled_path else []
            spec = {"mode": "type_sampled", "args": {"paths": paths, "n_items": 30}}

        return {
            "source": spec,
            "bin": args.bin_alias,
            "bin_payload": {
                "preview_cnt": int(args.preview_cnt),
                "max_steps_per_episode": int(args.max_steps),
            },
            "tag": f"{tag_prefix}_{args.mode}",
            "seed": ep_seed,
        }

    print(f"[debug] PalletFitEnv runner | episodes={args.episodes} mode={args.mode} "
          f"preview_cnt={args.preview_cnt} gif={args.gif}")
    print(f"[debug] obs space keys: {list(env.observation_space.spaces.keys())}")
    print(f"[debug] action space: {env.action_space}, NOOP_IDX={env.NOOP_IDX}")

    summary = []
    t_total = time.perf_counter()

    for ep in range(args.episodes):
        plan = _make_plan(ep)
        env.apply_plan(plan)

        try:
            obs, info = env.reset()
        except Exception as e:
            print(f"[ep {ep}] reset failed: {e}")
            traceback.print_exc()
            continue

        if info.get("invalid_episode"):
            print(f"[ep {ep}] invalid episode (items load failed): "
                  f"reason={info.get('terminal_reason')}")
            continue

        ep_return = 0.0
        steps = 0
        terminal_reason = None
        t_ep = time.perf_counter()

        for step_idx in range(args.max_steps + 5):
            # 마스크 기반 랜덤 액션 — NOOP은 가능한 다른 액션이 있으면 피한다.
            mask = np.asarray(env.action_masks(), dtype=bool)
            valid = np.where(mask)[0]
            if len(valid) == 0:
                # 이론상 _make_noop_only_state로 NOOP은 항상 살아 있어야 함
                action = env.NOOP_IDX
            else:
                non_noop = valid[valid != env.NOOP_IDX]
                action = int(rng.choice(non_noop)) if len(non_noop) > 0 else env.NOOP_IDX

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                print(f"[ep {ep} step {step_idx}] step crashed on action={action}: {e}")
                traceback.print_exc()
                terminal_reason = "exception"
                break

            ep_return += float(reward)
            steps += 1

            if terminated or truncated:
                terminal_reason = info.get("terminal_reason")
                break

        dt = time.perf_counter() - t_ep
        su = float(env.get_SU())
        packed = int(env._bin.size) if env._bin is not None else 0
        print(f"[ep {ep}] steps={steps:3d} return={ep_return:+.3f} "
              f"SU={su:.3f} packed={packed} reason={terminal_reason} "
              f"({dt*1000:.0f}ms, {dt*1000/max(1, steps):.1f}ms/step)")
        summary.append((ep, steps, ep_return, su, packed, terminal_reason))

    env.close()

    print("\n[debug] === summary ===")
    print(f"{'ep':>3} {'steps':>5} {'return':>8} {'SU':>6} {'packed':>6}  reason")
    for ep, steps, ep_return, su, packed, reason in summary:
        print(f"{ep:>3} {steps:>5} {ep_return:>+8.3f} {su:>6.3f} {packed:>6}  {reason}")
    if summary:
        sus = [r[3] for r in summary]
        print(f"[debug] mean SU = {sum(sus)/len(sus):.3f}  "
              f"total wall time = {(time.perf_counter()-t_total):.2f}s")
    
