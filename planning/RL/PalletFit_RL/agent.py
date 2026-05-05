# planning/RL/PalletFit_RL/agent.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
import os, datetime, shutil, csv
from pathlib import Path
import numpy as np
import torch as th

try:
    # 확률 합이 0.9999999 등이 나와도 에러내지 않도록 설정
    th.distributions.Distribution.set_default_validate_args(False)
    print("✅ PyTorch Distribution Validation disabled (Prevents Simplex Error)")
except Exception as e:
    print(f"⚠️ Failed to disable validation: {e}")

# =============================================================================
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

# SB3 / contrib
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor, VecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.logger import configure
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

# ── 우리의 환경 ──────────────────────────────────────────────
from planning.RL.PalletFit_RL.env import PalletFitEnv
from planning.RL.PalletFit_RL.custom_value_policy import PointerPolicyFT, CustomCombinedExtractor
from planning.RL.PalletFit_RL.config import ACTION_MAX_CANDIDATES, PREVIEW_MAX
from planning.data.data_sources import (
    spec_recorded, spec_type_sampled, spec_synthetic,
)
# =============================================================================
# 0) 설정
# =============================================================================

# 예시: 가장 잘 풀리고 데이터가 깔끔한 파일 하나 지정
OVERFIT_TARGET_PATH = "planning/data/Item_data/paper/testset/dataset_episode_012.json"


@dataclass
class EvalPolicy:
    """평가용 plan_maker 정책.

    train과 다른 분포에서 SU를 측정하면 학습 곡선 해석이 어려움
    EvalPolicy의 default는 **train mid phase와 같은 분포**로 설정 → train/eval SU 직접 비교 가능.

    deterministic 옵션(margin/preview/seed_offset)으로 재현성도 보장.

    Examples:
        # 기본 (train mid 분포 + deterministic)
        cfg = AgentConfig()

        # eval에서도 type_sampled/synthetic 비율 다르게
        cfg = AgentConfig(eval=EvalPolicy(p_type_sampled=0.5, p_synthetic=0.5))

        # eval만 다양한 margin으로 평가
        cfg = AgentConfig(eval=EvalPolicy(margin_x=4, margin_y=4))

        # eval만 큰 박스 (curriculum)
        cfg = AgentConfig(eval=EvalPolicy(synthetic_cfg=dict(min_item_mm=200, max_items=15)))
    """
    # 1) source 분포 (default = train mid phase와 동일) — Sec 2-8 핵심
    # 2026-05-05: 학습 분포 70/30 변경에 맞춰 평가 분포도 동일 적용
    p_scenario_fixed: float = 0.5
    p_type_sampled:   float = 0.7   # was 1.0
    p_synthetic:      float = 0.3   # was 0.0
    rehearsal_p:      float = 0.0   # eval은 rehearsal 안 함 (재현성)

    # 2) deterministic 옵션 — 매 평가마다 같은 plan을 보장
    margin_x: int = 0
    margin_y: int = 0
    preview_cnt: Optional[int] = None       # None이면 max(cfg.preview_cnt_choices)
    bin_alias: str = "experiment_RL"

    # 3) eval 전용 데이터 (None이면 cfg fallback)
    recorded_pool: Optional[List[str]] = None       # None → cfg.eval_recorded_paths_pool
    type_sampled_paths: Optional[List[str]] = None  # None → TYPE_SAMPLED_PATHS

    # 4) seed 분리 (학습 시드와 충돌 방지, 옛 hardcoded 100_000)
    seed_offset: int = 100_000


@dataclass
class BootstrapPolicy:
    """Bootstrap phase에서 강제하는 옵션들의 명시적 정책.

    학습 초기에 단순한 시나리오로 overfit시켜 의미 있는 행동을 빠르게 배우게 하는 단계.
    활성화되면 cfg의 일부 옵션(`type_sampled_ratio`, `synthetic_ratio`, `preview_cnt_choices`,
    `recorded_paths_pool` 등)이 silently 무시됨 → 어떤 게 강제되는지 한 곳에서 명시.

    기본값은 옛 동작과 동일 (학습 동작 무변화).

    Examples:
        # bootstrap 완전 끄기
        cfg = AgentConfig(bootstrap=BootstrapPolicy(enabled=False))
        # 다른 OVERFIT 파일
        cfg = AgentConfig(bootstrap=BootstrapPolicy(fixed_recorded_path="my.json"))
        # bootstrap에서도 type_sampled 허용
        cfg = AgentConfig(bootstrap=BootstrapPolicy(force_recorded_only=False))
    """
    enabled: bool = True
    duration_ratio: float = 0.10                                    # 전체 timesteps 대비 bootstrap 비율 (옛 bootstrap_ratio)
    fixed_recorded_path: Optional[str] = OVERFIT_TARGET_PATH        # 1개 파일로 고정 (None이면 정상 풀)
    fixed_preview_cnt: Optional[int] = None                         # None이면 PREVIEW_MAX 사용
    fixed_rollout_seed: bool = True                                 # rollout_idx=0 고정 (문제 순서 동일)
    force_recorded_only: bool = True                                # type_sampled/synthetic 비율을 0으로


@dataclass
class AgentConfig:
    device: str = "cuda" if th.cuda.is_available() else "cpu"
    seed: int = 1456
    total_timesteps: int = 20_000_000

    # ── PPO 핵심 ─────────────────────────────────────────────────
    # 2026-05-05: sparse reward(terminal_su) 적응 + 자원 활용 패키지 (Sec ?? 참조)
    # 하드웨어: Intel Xeon Gold 6526Y 16C16T (HT off), A6000 48GB, 125GiB RAM
    gamma: float = 0.999
    gae_lambda: float = 0.97             # ★ NEW: sparse reward → long-horizon credit 전파 (default 0.95에서 ↑)
    n_steps: int = 2048                  # was 1024 — n_envs 16으로 줄였으니 rollout 32k 유지
    batch_size: int = 8192               # was 4096 — A6000 추가 활용 (minibatches 8 → 4)
    n_epochs: int = 10                   # ★ NEW (cfg 노출, default SB3 동일)

    ent_coef: float = 0.05               # was 0.03 — sparse reward에서 exploration 보강
    vf_coef: float = 0.7                 # was 0.5  — terminal_su 큰 한 방 → critic 학습 비중 ↑
    clip_range: float = 0.2              # was 0.3  — sparse reward의 advantage variance 대응
    learning_rate: float = 3e-4

    # ── 벡터 환경 ────────────────────────────────────────────────
    # Option P: physical core 수와 정렬 (oversubscription 제거 → context-switch 비용 ↓)
    n_envs_train: int = 16               # was 32 — 16 physical cores와 1:1 매핑
    n_envs_eval: int = 8                 # was 5  — eval 통계 표본 ↑ (학습 시 16 + 평가 시 8 = 24 procs)
    # rollout = n_envs_train × n_steps = 16 × 2048 = 32,768 transitions per update
    # minibatches = 32768 / 8192 = 4

    # ── 주기(롤아웃 단위) 저장/평가 ────────────────────────────────
    # rollout이 32k로 커졌으니 같은 timesteps 도달까지 rollout 수가 1/10 → 주기 단축
    eval_every_rollouts: int = 2         # was 12 → 2 × 32768 = ~65k step마다 평가 (이전과 동일 빈도)
    save_every_rollouts: int = 6         # was 12 → 6 × 32768 = ~196k step마다 저장

    # ── 데이터 스케줄링 옵션 ─────────────────────────────────────
    # 1) Bootstrap phase 정책 — Sec 2-5 해결, BootstrapPolicy로 명시
    bootstrap: BootstrapPolicy = field(default_factory=BootstrapPolicy)
    # 1-b) Mid 페이즈 종료 비율 (mid 이후는 late)
    mid_ratio: float = 0.60
    # 1-c) Eval 정책 — Sec 2-8 해결, EvalPolicy로 명시 (default=train mid 분포)
    eval: EvalPolicy = field(default_factory=EvalPolicy)

    # 2) 리허설(recorded 고정 시나리오 유지학습) 비율: 페이즈별
    rehearsal_p_boot: float = 0.10
    rehearsal_p_mid:  float = 0.15
    rehearsal_p_late: float = 0.20
    # 3) 랜덤형 내부 비율 (2026-05-05: 70/30으로 조정)
    # - type_sampled 70%: real-world (real_box.json) 박스 분포 학습 (main)
    # - synthetic 30%: terminal_su=1.0 도달 가능 시나리오 → critic이 "최대치" V(s) 학습 (보조)
    # 실효 비율 (recorded 50% 가정): recorded 50% / type_sampled 35% / synthetic 15%
    type_sampled_ratio: float = 0.7   # was 1.0
    synthetic_ratio:    float = 0.3   # was 0.0

    # # recorded 리허설 완전 차단
    # rehearsal_p_boot=0.0
    # rehearsal_p_mid=0.0
    # rehearsal_p_late=0.0
    # # type_sampled:synthetic 비율을 0:1로
    # type_sampled_ratio=0.0
    # synthetic_ratio=1.0

    # 4) 성능 기반(Adaptive)도 켜고 싶을 때
    use_adaptive_phase: bool = False      # True면 성능기반 상향 허용(하이브리드)

    perf_window: int = 4                  # 최근 평가 W
    perf_eps_slope: float = 1e-3          # 증가량 임계
    perf_eps_std: float = 1e-2            # 분산 임계
    perf_min_eval_gap: int = 2            # 페이즈 전환 최소 평가 간격

    # ── 오프라인 소스 선택용 옵션 ──────────────────────────────
    # 사용자가 직접 경로 풀을 지정할 수 있게 (폴더/파일 혼합 가능)
    # recorded_paths_pool: Optional[List[str]] = None
    recorded_paths_pool=[
            "planning/data/Item_data/paper/setting123_discrete",
            "planning/data/Item_data/paper_drl/rs",
            'planning/data/Item_data/paper/testset'
        ]
    # [선택] 각 경로에 대한 가중치. None이면 자동으로 폴더 내 JSON 개수 비례로 계산
    # recorded_paths_weights: Optional[List[float]] = None
    recorded_paths_weights=[0.3, 0.3, 0.4]

    # 자동 가중치 계산을 켤지 여부 (weights가 None일 때만 의미 있음)
    recorded_auto_weight_by_len: bool = True
    # 평가용 오프라인 소스 (None이면 기본값 사용)
    # eval_recorded_paths_pool: Optional[List[str]] = None
    eval_recorded_paths_pool=[
            OVERFIT_TARGET_PATH,
            'planning/data/Item_data/paper/testset/dataset_episode_653.json',
            'planning/data/Item_data/paper/testset/dataset_episode_2731.json',
            'planning/data/Item_data/paper/testset/dataset_episode_1623.json',
            'planning/data/Item_data/paper/testset/dataset_episode_2999.json'
        ]
    # ── 액션 후보/선택 개수 튜닝 ────────────────────────────────
    preview_cnt_choices:     List[int] = field(default_factory=lambda: [1,2,3,4,5])


# =============================================================================
# Lite preset — 저사양 HW 재현 보호장치 (논문 reproducibility)
# =============================================================================
def make_lite_config() -> AgentConfig:
    """저사양 HW(4C/8T CPU + 6GB GPU + 16GB RAM)용 preset.

    학습 dynamics는 기본 cfg와 동일 — rollout 크기와 total_timesteps 보존.
    한 번에 처리하는 병렬도/배치만 줄여서 약한 머신에서 OOM 없이 돌게 함.
    수렴 속도(샘플 효율)는 동일, wall-clock만 더 걸림.

    사용:
        cfg = make_lite_config() if args.preset == "lite" else AgentConfig()
        agent = MaskablePPOAgent(cfg=cfg)
    """
    cfg = AgentConfig()
    cfg.n_envs_train = 4         # was 16  — 4 cores 가정
    cfg.n_envs_eval  = 2         # was 8
    cfg.n_steps      = 8192      # was 2048 — rollout = 4 × 8192 = 32,768 동일 유지
    cfg.batch_size   = 2048      # was 8192 — 6GB VRAM 안전 여유 (minibatches = 16)
    return cfg

# =============================================================================
# 1) Trainset Plan & Plan maker
# =============================================================================
@dataclass
class EnvPlan:
    """env에 전달되는 episode 계획.

    DataSource 추상화 (2026-05-04, docs/4 Sec 2-2 해결):
    이전엔 `item_mode`/`item_payload`로 mode별 분기를 표현했지만, 이제는 단일 `source`
    spec dict({"mode": ..., "args": ...})로 일원화. env는 spec → make_source().sample()
    한 줄로 처리.
    """
    # 에피소드 재현용 시드 (episode_seed로 들어감)
    seed: int

    # env 레벨 설정
    max_steps_per_episode: int = 200
    preview_cnt: int = PREVIEW_MAX

    # 메타
    mode: str = "train"         # "train" | "eval" (태그용)

    # 박스 source — DataSource spec dict
    # 예: {"mode": "synthetic", "args": {"max_items": 30, ...}}
    source: Dict[str, Any] = field(
        default_factory=lambda: {"mode": "recorded", "args": {"paths": []}}
    )

    # bin 쪽
    bin_alias: str = "experiment_RL"   # BinSpecsDict 키
    bin_margin_x: int = 0
    bin_margin_y: int = 0
    bin_payload: Dict[str, Any] = field(default_factory=dict)

# 공통 경로(프로젝트 상황에 맞게 수정)
RECORDED_PATHS = [
    "planning/data/Item_data/exhibition",
    "planning/data/Item_data/paper/setting123_discrete",
]
TYPE_SAMPLED_PATHS = [
"planning/data/Item_data/paper/real_box/real_box.json"
]
DEFAULT_SYNTHETIC_CFG = dict(
    bin_size=(1000, 1000, 1000),
    max_items=30,
    min_item_mm=100,
    max_aspect_ratio=3.0,
    margin_x=0,
    margin_y=0,
)

def env_plan_to_payload(p: EnvPlan) -> Dict[str, Any]:
    """EnvPlan → env.apply_plan에 전달할 dict.

    새 canonical 형식(2026-05-04, DataSource 추상화):
        {
            "source":      {"mode": ..., "args": {...}},
            "bin":         alias,
            "bin_payload": {margin_x, margin_y, preview_cnt, max_steps_per_episode},
            "tag":         "train" | "eval_*",
            "seed":        episode seed,
        }
    """
    bin_payload = dict(p.bin_payload or {})
    bin_payload.setdefault("max_steps_per_episode", int(p.max_steps_per_episode))
    bin_payload.setdefault("margin_x", int(p.bin_margin_x))
    bin_payload.setdefault("margin_y", int(p.bin_margin_y))
    bin_payload.setdefault("preview_cnt", int(p.preview_cnt))

    return {
        "source":      dict(p.source or {}),
        "bin":         p.bin_alias,
        "bin_payload": bin_payload,
        "tag":         str(p.mode),
        "seed":        int(p.seed),
    }

def _phase_from_steps(current_steps: int, total_timesteps: int,
                      bootstrap_duration_ratio: float, mid_ratio: float) -> str:
    """시간(샘플) 기반 페이즈 판정.

    bootstrap_duration_ratio: 전체 timesteps 대비 bootstrap 차지 비율
        (BootstrapPolicy.duration_ratio에서 옴).
    """
    b = int(total_timesteps * bootstrap_duration_ratio)
    m = int(total_timesteps * mid_ratio)
    if current_steps < b:
        return "bootstrap"  # 70:30
    elif current_steps < m:
        return "mid"        # 50:50
    else:
        return "late"       # 40:60

def _rehearsal_p_from_phase(phase: str, cfg: AgentConfig) -> float:
    return {
        "bootstrap": cfg.rehearsal_p_boot,
        "mid":       cfg.rehearsal_p_mid,
        "late":      cfg.rehearsal_p_late,
    }[phase]

def _mix_ratio_from_phase(phase: str) -> tuple[float, float]:
    """시나리오 고정형 vs 랜덤형 비율"""
    if phase == "bootstrap":
        return 0.70, 0.30
    elif phase == "mid":
        return 0.50, 0.50
    else:
        return 0.40, 0.60
    
def _count_json_episodes(p: str) -> int:
    """폴더면 *.json 개수, 파일이면 확장자 json이면 1, 아니면 0"""
    path = Path(p)
    if path.is_dir():
        return len(list(path.glob("*.json")))
    if path.is_file() and path.suffix.lower() == ".json":
        return 1
    return 0

def _normalize_probs(xs: List[float]) -> List[float]:
    s = float(sum(max(0.0, x) for x in xs))
    if s <= 0:
        n = len(xs)
        return [1.0 / n] * n if n > 0 else []
    return [max(0.0, x) / s for x in xs]

def _resolve_recorded_pool_and_weights(cfg: AgentConfig) -> tuple[List[str], Optional[List[float]]]:
    """
    cfg에서 오프라인 풀/가중치를 가져오되, 없으면 기본 RECORDED_PATHS 사용.
    가중치가 없고 recorded_auto_weight_by_len=True면 폴더 내 json 개수로 가중치 자동 계산.
    """
    pool = list(cfg.recorded_paths_pool) if cfg.recorded_paths_pool else list(RECORDED_PATHS)
    if not pool:
        raise ValueError("No recorded_paths available")

    if cfg.recorded_paths_weights is not None:
        # 사용자가 명시한 가중치
        weights = list(cfg.recorded_paths_weights)
        if len(weights) != len(pool):
            raise ValueError("recorded_paths_weights length must match recorded_paths_pool")
        return pool, weights

    if cfg.recorded_auto_weight_by_len:
        counts = [max(1, _count_json_episodes(p)) for p in pool]  # 최소 1 보장
        return pool, counts

    # 둘 다 아니면 균등
    return pool, None

# ---------- (A) 유틸: 폴더/파일 풀을 파일 리스트로 전개 ----------
def _list_json_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_dir():
        return [str(x) for x in sorted(p.glob("*.json"))]
    if p.is_file() and p.suffix.lower() == ".json":
        return [str(p)]
    return []

def _expand_recorded_pool_to_files(pool: List[str]) -> List[str]:
    files: List[str] = []
    for p in pool:
        files.extend(_list_json_files(p))
    # 중복 제거(같은 파일이 여러 경로에서 들어올 수 있으니)
    return sorted(list(dict.fromkeys(files)))

# ---------- (B) 폴더 가중치를 파일 가중치로 풀어주기 ----------
def _distribute_folder_weights_to_files(pool: List[str], folder_weights: Optional[List[float]]) -> Optional[List[float]]:
    files = _expand_recorded_pool_to_files(pool)
    if not files:
        return []

    if folder_weights is None:
        return None  # 균등 샘플링

    # 폴더/파일 혼재: 파일이 속한 폴더 weight를 그 파일들에게 균등 분배
    folder_to_w = {str(Path(p)): float(w) for p, w in zip(pool, folder_weights)}
    file_weights: List[float] = []
    # 폴더별 파일 개수 미리 계산
    folder_to_files = defaultdict(list)
    for f in files:
        folder_to_files[str(Path(f).parent)].append(f)

    for f in files:
        folder = str(Path(f).parent)
        fw = float(folder_to_w.get(folder, 0.0))
        cnt = max(1, len(folder_to_files[folder]))
        file_weights.append(fw / cnt)
    return file_weights

# ---------- (C) 히스토리 기반 파일 선택 ----------
def _weighted_choice(files: List[str], file_weights: Optional[List[float]]) -> str:
    """파일 가중 sample. weights None이면 균등."""
    if file_weights is None:
        return str(np.random.choice(files))
    probs = _normalize_probs(file_weights)
    return files[int(np.random.choice(len(files), p=probs))]


def _choose_unseen_first(
    files: List[str],
    file_weights: Optional[List[float]],
    history: "DatasetHistory",
) -> Optional[str]:
    """안 배운 파일 우선 선택 (탐험). 평소 학습용."""
    if not files:
        return None
    # 아직 한 번도 안 배운 파일들
    unseen = [f for f in files if history.recorded.get(f, SourceStat()).count == 0]
    if unseen:
        if file_weights is None:
            return str(np.random.choice(unseen))
        # 부분집합에 대한 가중치 재정규화
        idxs = [files.index(u) for u in unseen]
        subw = _normalize_probs([file_weights[i] for i in idxs])
        return unseen[int(np.random.choice(len(unseen), p=subw))]
    # 전부 배운 적 있으면 가중치/균등으로
    return _weighted_choice(files, file_weights)


def _choose_mastered_first(
    files: List[str],
    file_weights: Optional[List[float]],
    history: "DatasetHistory",
) -> Optional[str]:
    """이미 mastered한 파일 우선 선택 (catastrophic forgetting 방지 = 진짜 rehearsal).

    DatasetHistory.mastered_recorded()의 임계값(min_episodes_master / su_master_threshold)
    기준으로 mastered 판정.
    학습 초기엔 mastered=∅이라 fallback으로 일반 가중 sample (fixed 분기와 동일 동작).

    docs/4 Sec 2-4 해결 (2026-05-04): 옛 _choose_recorded_file_with_history는 동작이 "탐험"인데
    이름이 "복습"이라 의미 충돌. 이제 unseen-first(탐험) / mastered-first(복습) 두 함수로 분리하고
    plan_maker에서 explore vs rehearsal 분기에 각각 적용.
    """
    if not files:
        return None
    mastered_set = set(history.mastered_recorded())
    mastered = [f for f in files if f in mastered_set]
    if mastered:
        if file_weights is None:
            return str(np.random.choice(mastered))
        idxs = [files.index(m) for m in mastered]
        subw = _normalize_probs([file_weights[i] for i in idxs])
        return mastered[int(np.random.choice(len(mastered), p=subw))]
    # 아직 mastered 없음 → 평소 가중치/균등 sample (학습 초기 fallback)
    return _weighted_choice(files, file_weights)

def _make_plans_core(
    seed: int,
    n_envs: int,
    *,
    p_scenario_fixed: float,
    p_type_sampled: float,
    p_synthetic: float,          
    rehearsal_p: float,
    rollout_idx: int = 0,
    bin_key: str = "experiment_RL",
    margin_range: tuple[int, int] = (0, 8),
    recorded_file_candidates: Optional[List[str]] = None,
    recorded_file_weights: Optional[List[float]] = None,
    history: Optional["DatasetHistory"] = None,
    preview_cnt_choices: Optional[List[int]] = None,
    ) -> List[EnvPlan]:
    
    plans: List[EnvPlan] = []
    files = list(recorded_file_candidates or [])

    # [수정] 기본 후보군 설정 (없으면 PREVIEW_MAX 하나만 사용)
    pc_choices = list(preview_cnt_choices or [PREVIEW_MAX])

    for i in range(n_envs):
        # 1) preview_cnt 샘플링 (하나로 통합)
        p_cnt = int(np.random.choice(pc_choices))
        
        # 안전 장치: 1 ~ PREVIEW_MAX 사이로 클램핑
        p_cnt = max(1, min(p_cnt, PREVIEW_MAX))

        # 2) margin 샘플 (기존 동일)
        mx = int(np.random.randint(*margin_range))
        my = int(np.random.randint(*margin_range))

        # 3) 이 env 에피소드용 시드 (기존 동일)
        ep_seed = int(seed + rollout_idx * 10_000 + i)

        # 공통 플랜 생성기 — DataSource spec dict + bin/margin/preview 묶기
        def _make_plan(spec: Dict[str, Any], tag: str) -> EnvPlan:
            return EnvPlan(
                seed=ep_seed,
                max_steps_per_episode=200,
                preview_cnt=p_cnt,
                mode=tag,
                source=spec,
                bin_alias=bin_key,
                bin_margin_x=mx,
                bin_margin_y=my,
                bin_payload={},
            )

        # 4) 진짜 rehearsal — mastered한 source 우선 (catastrophic forgetting 방지)
        if np.random.rand() < rehearsal_p:
            chosen = (_choose_mastered_first(files, recorded_file_weights, history)
                      if history else
                      (files[int(np.random.choice(len(files)))] if files else None))
            plans.append(_make_plan(
                spec_recorded([chosen] if chosen else []),
                tag="train-rehearsal",
            ))
            continue

        # 5) 고정형(평소 학습): 안 배운 source 우선 (탐험)
        if np.random.rand() < p_scenario_fixed:
            chosen = (_choose_unseen_first(files, recorded_file_weights, history)
                      if history else
                      (files[int(np.random.choice(len(files)))] if files else None))
            plans.append(_make_plan(
                spec_recorded([chosen] if chosen else []),
                tag="train",
            ))
        else:
            # 랜덤형: type_sampled vs synthetic
            if np.random.rand() < p_type_sampled:
                plans.append(_make_plan(
                    spec_type_sampled(TYPE_SAMPLED_PATHS),
                    tag="train",
                ))
            else:
                plans.append(_make_plan(
                    spec_synthetic(**DEFAULT_SYNTHETIC_CFG),
                    tag="train",
                ))

    return plans

def make_time_based_plan_maker(cfg: AgentConfig, history: "DatasetHistory"):
    def plan_maker(rollout_idx: int, n_envs: int, *, current_steps: int) -> List[EnvPlan]:
        # 1. 현재 페이즈 확인 (bootstrap.duration_ratio 사용)
        phase = _phase_from_steps(
            current_steps, cfg.total_timesteps,
            cfg.bootstrap.duration_ratio, cfg.mid_ratio,
        )

        p_scenario_fixed, _ = _mix_ratio_from_phase(phase)
        rehearsal_p = _rehearsal_p_from_phase(phase, cfg)

        # 정상 풀 (mid/late 또는 bootstrap.enabled=False일 때)
        recorded_pool, folder_weights = _resolve_recorded_pool_and_weights(cfg)
        recorded_files = _expand_recorded_pool_to_files(recorded_pool)
        recorded_file_weights = _distribute_folder_weights_to_files(recorded_pool, folder_weights)
        cur_pc_choices = list(cfg.preview_cnt_choices)
        final_rollout_idx = rollout_idx
        real_p_fixed = p_scenario_fixed
        real_p_type_sampled = cfg.type_sampled_ratio
        real_p_synthetic = cfg.synthetic_ratio

        # 2. Bootstrap phase 강제 옵션 적용 — BootstrapPolicy 항목별 토글
        bs = cfg.bootstrap
        if bs.enabled and phase == "bootstrap":
            if bs.fixed_recorded_path:
                recorded_files = [bs.fixed_recorded_path]
                recorded_file_weights = None
            if bs.force_recorded_only:
                real_p_fixed = 1.0
                real_p_type_sampled = 0.0
                real_p_synthetic = 0.0
            if bs.fixed_rollout_seed:
                final_rollout_idx = 0
            if bs.fixed_preview_cnt is not None:
                cur_pc_choices = [int(bs.fixed_preview_cnt)]
            else:
                # 기본: PREVIEW_MAX 하나로 고정 (가장 정보 많은 상태에서 학습)
                cur_pc_choices = [PREVIEW_MAX]

        return _make_plans_core(
            cfg.seed,     # positional arg 1
            n_envs,       # positional arg 2

            p_scenario_fixed=real_p_fixed,
            p_type_sampled=real_p_type_sampled,
            p_synthetic=real_p_synthetic,
            rehearsal_p=rehearsal_p,
            
            rollout_idx=final_rollout_idx,  
            
            recorded_file_candidates=recorded_files,
            recorded_file_weights=recorded_file_weights,
            history=history,
            
            # [수정] 통합된 choices 전달
            preview_cnt_choices=cur_pc_choices,
        )
    return plan_maker

def make_eval_plan_maker(cfg: AgentConfig) -> Callable[[int, int], List[EnvPlan]]:
    """Eval plan_maker — `_make_plans_core` 재사용으로 train과 동일 코드 경로.

    docs/4 Sec 2-8 해결 (2026-05-05): 옛날엔 별도 200줄 직렬 loop라 train과 분포가
    어긋났음. 이제 EvalPolicy로 정책 명시 → train과 같은 함수에 인자만 다르게.
    """
    ep = cfg.eval

    # 1) recorded pool 결정 (EvalPolicy override > cfg.eval_recorded_paths_pool > cfg.recorded_paths_pool)
    if ep.recorded_pool is not None:
        recorded_pool = list(ep.recorded_pool)
    elif cfg.eval_recorded_paths_pool:
        recorded_pool = list(cfg.eval_recorded_paths_pool)
    elif cfg.recorded_paths_pool:
        recorded_pool = list(cfg.recorded_paths_pool)
    else:
        recorded_pool = []
    recorded_files = _expand_recorded_pool_to_files(recorded_pool) if recorded_pool else []

    # 2) preview_cnt 결정 (EvalPolicy override > max(cfg.preview_cnt_choices) > PREVIEW_MAX)
    if ep.preview_cnt is not None:
        preview_cnt_eval = int(ep.preview_cnt)
    elif cfg.preview_cnt_choices:
        preview_cnt_eval = int(max(cfg.preview_cnt_choices))
    else:
        preview_cnt_eval = int(PREVIEW_MAX)

    def plan_maker_eval(rollout_idx: int, n_envs: int, *, current_steps: int) -> List[EnvPlan]:
        # train의 _make_plans_core 그대로 재사용 — 동일 코드 경로!
        # eval 차이점은 인자로만 표현:
        #   - history=None      → rehearsal/explore 비활성 (deterministic)
        #   - margin_range=(m, m+1) → 단일값 강제 (deterministic margin)
        #   - preview_cnt_choices=[ep.preview_cnt] → 단일값 강제
        #   - seed offset 분리
        plans = _make_plans_core(
            cfg.seed + ep.seed_offset,
            n_envs,
            p_scenario_fixed=ep.p_scenario_fixed,
            p_type_sampled=ep.p_type_sampled,
            p_synthetic=ep.p_synthetic,
            rehearsal_p=ep.rehearsal_p,
            rollout_idx=rollout_idx,
            bin_key=ep.bin_alias,
            margin_range=(ep.margin_x, ep.margin_x + 1),
            recorded_file_candidates=recorded_files,
            recorded_file_weights=None,
            history=None,
            preview_cnt_choices=[preview_cnt_eval],
        )
        # tag 재라벨 (train → eval_*) — 분석/로그/GIF 캡처용
        for p in plans:
            mode = p.source.get("mode", "unknown")
            p.mode = f"eval_{mode}"
        return plans

    return plan_maker_eval

# =============================================================================
# 2) Dataset History
# ============================================================================
@dataclass
class SourceStat:
    count: int = 0
    ema_su: float = 0.0      # SU의 EMA
    last_su: float = 0.0
    last_ts: int = 0

@dataclass
class DatasetHistory:
    recorded:     Dict[str, SourceStat] = field(default_factory=lambda: defaultdict(SourceStat))
    type_sampled: Dict[str, SourceStat] = field(default_factory=lambda: defaultdict(SourceStat))
    synthetic:    Dict[str, SourceStat] = field(default_factory=lambda: defaultdict(SourceStat))

    ema_alpha: float = 0.2
    min_episodes_master: int = 10
    su_master_threshold: float = 0.85

    def _bucket(self, mode: str):
        return {"recorded": self.recorded, "type_sampled": self.type_sampled, "synthetic": self.synthetic}[mode]

    def record(
        self,
        *,
        source_mode: str,
        source_id: str,
        su: float,
        timesteps: int,
    ) -> None:
        """env가 info에 노출한 (source_mode, source_id) 기반 통계 누적.

        DataSource 추상화 (2026-05-04, docs/4 Sec 2-2/2-7 해결):
        - mode별 분기 제거 — bucket lookup만.
        - source_id 입자가 source 클래스에서 결정 (synthetic도 cfg variant 별 분리).
        """
        try:
            bucket = self._bucket(str(source_mode))
        except KeyError:
            return  # 모르는 모드는 무시
        st = bucket[str(source_id)]
        st.count += 1
        st.last_su = float(su)
        st.ema_su = (1.0 - self.ema_alpha) * st.ema_su + self.ema_alpha * float(su)
        st.last_ts = int(timesteps)

    def record_from_info(self, info: Dict[str, Any], su: float, timesteps: int) -> None:
        """env._build_info의 source_mode/source_id로 바로 기록."""
        if not isinstance(info, dict):
            return
        mode = info.get("source_mode")
        sid = info.get("source_id")
        if not mode or not sid:
            return
        self.record(source_mode=mode, source_id=sid, su=su, timesteps=timesteps)
    # ---- (옵션) 어느 recorded source가 '마스터' 되었는지 판단 ----
    def mastered_recorded(self) -> List[str]:
        """
        충분히 많이 등장했고(SU가 안정적으로 높은) recorded 소스들 리스트.
        지금은 코드 어디에서도 안 쓰지만, 나중에 curriculum 짤 때 유용할 수 있음.
        """
        out: List[str] = []
        for src, st in self.recorded.items():
            if st.count >= self.min_episodes_master and st.ema_su >= self.su_master_threshold:
                out.append(src)
        return out

    # ---- 디버깅/텍스트 출력용 스냅샷 ----
    def debug_snapshot(self) -> Dict[str, Dict[str, dict]]:
        """
        {mode: {source_id: {count, ema_su, last_su, last_ts}}}
        형태로 현재 히스토리 상태를 리턴.
        """
        def pack(d: Dict[str, SourceStat]) -> Dict[str, dict]:
            return {
                src: {
                    "count": st.count,
                    "ema_su": float(st.ema_su),
                    "last_su": float(st.last_su),
                    "last_ts": int(st.last_ts),
                }
                for src, st in d.items()
            }

        return {
            "recorded": pack(self.recorded),
            "type_sampled": pack(self.type_sampled),
            "synthetic": pack(self.synthetic),
        }

    # ---- CSV 저장용 row 리스트 만들기 ----
    def to_rows(self) -> List[Dict[str, Any]]:
        """
        CSV로 저장하기 좋은 형태의 리스트를 리턴.
        각 row는 {"mode","source","count","ema_su","last_su","last_ts"} 필드를 가짐.
        """
        rows: List[Dict[str, Any]] = []

        def dump(mode: str, d: Dict[str, SourceStat]) -> None:
            for src, st in d.items():
                rows.append({
                    "mode": mode,
                    "source": src,
                    "count": st.count,
                    "ema_su": float(st.ema_su),
                    "last_su": float(st.last_su),
                    "last_ts": int(st.last_ts),
                })

        dump("recorded", self.recorded)
        dump("type_sampled", self.type_sampled)
        dump("synthetic", self.synthetic)
        return rows

    # ---- CSV로 저장 ----
    def save_csv(self, path: str) -> None:
        """
        dataset_history.csv로 저장.
        - mode, source, count, ema_su, last_su, last_ts
        """
        rows = self.to_rows()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        header = ["mode", "source", "count", "ema_su", "last_su", "last_ts"]
        with path_obj.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                # source 안에 콤마가 있으면 CSV 깨지니까 공백으로 치환
                src = str(r["source"]).replace(",", " ")
                line = ",".join([
                    str(r["mode"]),
                    src,
                    str(int(r["count"])),
                    f"{float(r['ema_su']):.6f}",
                    f"{float(r['last_su']):.6f}",
                    str(int(r["last_ts"])),
                ])
                f.write(line + "\n")

    # ---- 사람이 읽기 쉬운 텍스트로 저장 ----
    def save_txt(self, path: str) -> None:
        """
        dataset_history.txt로 저장.
        모드별로 나눠서 각 source의 통계를 사람이 읽기 쉽게 출력.
        """
        snap = self.debug_snapshot()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with path_obj.open("w", encoding="utf-8") as f:
            for mode in ["recorded", "type_sampled", "synthetic"]:
                f.write(f"== {mode} ==\n")
                mdict = snap.get(mode, {})
                for src, st in mdict.items():
                    f.write(
                        f"- {src}: "
                        f"count={st['count']} "
                        f"ema_su={st['ema_su']:.4f} "
                        f"last_su={st['last_su']:.4f} "
                        f"last_ts={st['last_ts']}\n"
                    )
                f.write("\n")

    # 옛 mode 이름 → 새 이름 매핑 (옛 dataset_history.csv 호환)
    _MODE_LEGACY_ALIAS = {
        "offline":     "recorded",
        "online_type": "type_sampled",
        "tsg":         "synthetic",
    }

    def load_csv(self, path: str):
        """
        이전 run에서 저장한 dataset_history.csv를 읽어서
        recorded / type_sampled / synthetic 통계를 복구한다.
        옛 이름(offline/online_type/tsg)도 자동으로 새 이름으로 매핑.
        """
        path = Path(path)
        if not path.is_file():
            print(f"[DatasetHistory] CSV not found, skip load: {path}")
            return

        # 기존 내용 초기화
        self.recorded     = defaultdict(SourceStat)
        self.type_sampled = defaultdict(SourceStat)
        self.synthetic    = defaultdict(SourceStat)

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mode   = row.get("mode")
                source = row.get("source")
                if not mode or not source:
                    continue

                # 옛 이름이면 새 이름으로 변환
                mode = self._MODE_LEGACY_ALIAS.get(mode, mode)

                try:
                    bucket = self._bucket(mode)
                except (KeyError, ValueError):
                    # 모르는 mode면 무시
                    continue

                st = bucket[source]  # defaultdict라서 자동 생성
                try:
                    st.count   = int(row.get("count", "0"))
                except ValueError:
                    pass
                try:
                    st.ema_su  = float(row.get("ema_su", "0.0"))
                except ValueError:
                    pass
                try:
                    st.last_su = float(row.get("last_su", "0.0"))
                except ValueError:
                    pass
                try:
                    st.last_ts = int(row.get("last_ts", "0"))
                except ValueError:
                    pass

        print(f"[DatasetHistory] Loaded history from CSV: {path}")

# =============================================================================
# 2) 성능 기반(옵션) 페이즈 컨트롤러
#    - 시간 기반과 하이브리드로 사용 권장
# =============================================================================
class AdaptivePhaseController:
    def __init__(self, *, W=4, eps_slope=1e-3, eps_std=1e-2, min_eval_gap=2):
        self.phase = 0  # 0:bootstrap, 1:mid, 2:late
        self.hist: List[float] = []
        self.W = int(W)
        self.eps_slope = float(eps_slope)
        self.eps_std = float(eps_std)
        self.min_eval_gap = int(min_eval_gap)
        self._last_phase_eval_idx = -10
        self._eval_count = 0

    def _phase_to_ratios(self):
        return [(0.70, 0.30), (0.50, 0.50), (0.40, 0.60)][self.phase]

    def _phase_to_rehearsal(self):
        return [0.10, 0.15, 0.20][self.phase]

    def on_eval(self, su_mean: float) -> None:
        self._eval_count += 1
        self.hist.append(float(su_mean))
        self.hist = self.hist[-(self.W + 2):]

        if (self._eval_count - self._last_phase_eval_idx) < self.min_eval_gap:
            return
        if len(self.hist) < self.W:
            return
        y = np.array(self.hist[-self.W:])
        slope = (y[-1] - y[0]) / max(1, self.W - 1)
        stable = (abs(slope) < self.eps_slope) and (np.std(y) < self.eps_std)
        if stable and self.phase < 2:
            self.phase += 1
            self._last_phase_eval_idx = self._eval_count

    def ratios(self):
        return self._phase_to_ratios()

    def rehearsal_p(self):
        return self._phase_to_rehearsal()


# =============================================================================
# 3) 콜백: TrainsetScheduling (per-rollout)
# =============================================================================
class TrainsetSchedulingCallback(BaseCallback):
    def __init__(self, plan_maker, verbose: int = 0):
        super().__init__(verbose)
        self.plan_maker = plan_maker
        self.rollout_idx = 0

    def _on_rollout_start(self) -> None:
        venv = self.model.get_env()
        n = int(getattr(venv, "num_envs", 1))
        current_steps = int(getattr(self.model, "num_timesteps", 0))

        plans: List[EnvPlan] = self.plan_maker(self.rollout_idx, n, current_steps=current_steps)
        assert len(plans) == n

        for i, p in enumerate(plans):
            payload = env_plan_to_payload(p)
            venv.env_method("apply_plan", payload, indices=[i])

        self.rollout_idx += 1

    def _on_step(self) -> bool:
        return True
# =============================================================================
# 4) 간단 로깅 + 평가-적응 하이브리드 콜백
# =============================================================================
class EvalAndAdaptiveCallback(BaseCallback):
    def __init__(
        self, 
        *, 
        eval_fn, 
        eval_interval_steps: int,
        adaptive_ctl=None, 
        inject_adaptive_to_plan_maker=None, 
        best_model_save_path: str | None = None,  # [추가] 저장 경로
        verbose=0
    ):
        super().__init__(verbose)
        self.eval_fn = eval_fn
        self.eval_interval_steps = int(eval_interval_steps)
        self.adaptive_ctl = adaptive_ctl
        self.inject_adaptive_to_plan_maker = inject_adaptive_to_plan_maker
        self._last_eval_ts = 0
        
        # [추가] Best Model 추적용 변수 및 파일 경로 설정
        self.best_model_save_path = best_model_save_path
        self.best_su = -float('inf')
        self.best_su_file = None

        # 파일에서 기존 Best SU 불러오기 (Persistence)
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
            self.best_su_file = os.path.join(self.best_model_save_path, "best_su.txt")
            
            if os.path.exists(self.best_su_file):
                try:
                    with open(self.best_su_file, "r") as f:
                        val = float(f.read().strip())
                        self.best_su = val
                    if self.verbose > 0:
                        print(f"[EvalCallback] Loaded previous best SU: {self.best_su:.4f}")
                except Exception as e:
                    print(f"[EvalCallback] Warning: Failed to load best_su.txt: {e}")

    def _on_step(self) -> bool:
        ns = int(self.num_timesteps)
        # 마지막 평가로부터 일정 스텝이 지났는지 확인
        if (ns - self._last_eval_ts) < self.eval_interval_steps:
            return True
        self._last_eval_ts = ns

        # 1. 평가 수행
        metrics = self.eval_fn() or {}
        
        if metrics:
            # 2. Best Model 저장 로직 (파일 Persistence 포함)
            current_su = float(metrics.get("SU", -float('inf')))
            
            # 기존 기록보다 더 좋을 때만 갱신
            if current_su > self.best_su:
                self.best_su = current_su
                
                if self.best_model_save_path is not None:
                    # (A) 모델 저장
                    save_path = os.path.join(self.best_model_save_path, "best_model")
                    self.model.save(save_path)
                    
                    # (B) 점수 파일 저장 (다음 재시작을 위해)
                    try:
                        with open(self.best_su_file, "w") as f:
                            f.write(str(self.best_su))
                    except Exception as e:
                        print(f"[EvalCallback] Warning: Failed to write best_su.txt: {e}")

                    if self.verbose > 0:
                        print(f"🔥 New best model saved to {save_path}.zip (SU={current_su:.4f})")

            # 3. SB3 logger 기록
            for k, v in metrics.items():
                self.logger.record(f"eval/{k}", float(v))
            self.logger.dump(ns)

        # 4. 어댑티브 제어 (기존 로직 유지)
        if self.adaptive_ctl is not None and metrics:
            su_mean = float(metrics.get("SU_mean", metrics.get("SU", 0.0))) # SU 또는 SU_mean 사용
            self.adaptive_ctl.on_eval(su_mean)
            p_fixed, _ = self.adaptive_ctl.ratios()
            rehearsal_p = self.adaptive_ctl.rehearsal_p()
            if self.inject_adaptive_to_plan_maker is not None:
                self.inject_adaptive_to_plan_maker(self.adaptive_ctl.phase, p_fixed, rehearsal_p)
        
        return True

# =============================================================================
# 5) dataset history 콜백
# =============================================================================
class HistoryCollectorCallback(BaseCallback):
    def __init__(self, history: DatasetHistory, ema_alpha: float = 0.2, verbose: int = 0):
        super().__init__(verbose)
        self.history = history
        self.ema_alpha = float(ema_alpha)
        self._su_ema: float = 0.0
        self._su_count: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None or dones is None:
            return True

        ts = int(self.num_timesteps)
        any_done = False

        for done, info in zip(dones, infos):
            if not bool(done) or not isinstance(info, dict):
                continue

            any_done = True

            # ── env info에서 값 꺼내기 ───────────────────
            su           = float(info.get("SU", 0.0))
            ep_ret       = float(info.get("return", 0.0))
            steps_in_ep  = int(info.get("steps_in_ep", 0))
            packed_cnt   = int(info.get("packed_count", 0))
            mode         = str(info.get("source_mode", "unknown"))

            # [수정] 통합된 변수
            preview_cnt  = int(info.get("preview_cnt", 0))

            # ── DatasetHistory 적립 (info의 source_mode/source_id 직접 사용) ─
            self.history.record_from_info(info, su=su, timesteps=ts)

            # ── SU running EMA 업데이트 ─────────────────
            self._su_count += 1
            if self._su_count == 1:
                self._su_ema = su
            else:
                self._su_ema = (1.0 - self.ema_alpha) * self._su_ema + self.ema_alpha * su

            # ── TensorBoard train/* 로깅 (에피소드 단위) ─
            self.logger.record(f"train/{mode}_SU", su)
            self.logger.record(f"train/{mode}_packed", packed_cnt)
            self.logger.record(f"train/{mode}_return", ep_ret)            
            self.logger.record(f"train/{mode}_episode_len", steps_in_ep)
            
            # [수정] 깔끔하게 preview_cnt 하나만 남김
            self.logger.record(f"train/{mode}_preview_cnt", preview_cnt)
            
            # (삭제됨: head_slots 등)

            for key, val in info.items():
                if key.startswith("r_ep_"):
                    name = key[len("r_ep_"):]  
                    self.logger.record(f"train/reward_ep/{name}", float(val))

        if any_done:
            self.logger.dump(ts)

        return True


class HistorySaverCallback(BaseCallback):
    def __init__(self, history: "DatasetHistory", out_dir: str, save_every_steps: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.history = history
        self.out_dir = Path(out_dir)
        self.save_every_steps = int(save_every_steps)
        self._last_save_ts = 0

    def _save_now(self, ts: int):
        csv_path = self.out_dir / "dataset_history.csv"
        txt_path = self.out_dir / "dataset_history.txt"
        self.history.save_csv(str(csv_path))
        self.history.save_txt(str(txt_path))
        if self.verbose:
            print(f"[HistorySaver] saved history at {ts} steps -> {csv_path.name}, {txt_path.name}")

    def _on_step(self) -> bool:
        ts = int(self.num_timesteps)
        if (ts - self._last_save_ts) >= self.save_every_steps:
            self._last_save_ts = ts
            self._save_now(ts)
        return True

    def _on_training_end(self) -> None:
        self._save_now(int(self.num_timesteps))


# =============================================================================
# 5) Env 팩토리
# =============================================================================
info_keys = ("SU","steps_in_ep","return","packed_count",)

def make_single_env(seed: int, tb_log_dir: str, *, is_render_env: bool = False):
    def _thunk():
        env = PalletFitEnv(seed=seed, tb_log_dir=tb_log_dir, is_render_env=is_render_env)
        return Monitor(env, info_keywords=info_keys)
    return _thunk

def make_envs(*, mode: str, n_envs: int, base_seed: int, tb_log_dir: str) -> VecEnv:
    thunks = [make_single_env(base_seed + i, tb_log_dir) for i in range(n_envs)]
    venv: VecEnv = SubprocVecEnv(thunks, start_method="spawn")
    venv = VecMonitor(venv, filename=str(Path(tb_log_dir) / f"monitor_{mode}"))
    return venv

def make_eval_env(*, n_envs: int, base_seed: int, tb_log_dir: str) -> VecEnv:
    """평가용 SubprocVecEnv. 0번 워커만 GIF 저장(is_render_env=True)."""
    thunks = [
        make_single_env(base_seed + i, tb_log_dir, is_render_env=(i == 0))
        for i in range(n_envs)
    ]
    venv: VecEnv = SubprocVecEnv(thunks, start_method="spawn")
    venv = VecMonitor(venv, filename=str(Path(tb_log_dir) / "monitor_eval"))
    return venv


# =============================================================================
# 6) Agent
# =============================================================================
class MaskablePPOAgent:
    def __init__(self, *, cfg: Optional[AgentConfig] = None, override_log_dir: Optional[str] = None):        
        self.cfg = cfg or AgentConfig()
        
        if override_log_dir:
            # 1) 외부에서 지정한 고정 경로 사용 (무한 루프용)
            self.log_dir = Path(override_log_dir)
            print(f"[Agent] Using overridden log_dir: {self.log_dir}")
        else:
            # 2) 날짜 기반 경로 생성 (단발성 실행용)
            self.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.logs_root = Path("planning/RL/PalletFit_RL/logs")
            self.log_dir = self.logs_root / f"MaskablePPO_{self.run_id}"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = DatasetHistory()
        self._cached_eval_plans: Optional[List[EnvPlan]] = None

        # 1) VecEnvs 먼저
        self.env = make_envs(
            mode="train",
            n_envs=self.cfg.n_envs_train,
            base_seed=self.cfg.seed,
            tb_log_dir=str(self.log_dir),
        )

        # 평가 환경: SubprocVecEnv 병렬, 0번 워커는 GIF 저장
        self.eval_env = make_eval_env(
            n_envs=int(self.cfg.n_envs_eval),
            base_seed=self.cfg.seed + 10_000,
            tb_log_dir=str(self.log_dir),
        )

        # 2) 모델 생성
        self.model = MaskablePPO(
            policy=PointerPolicyFT,
            env=self.env,
            learning_rate=self.cfg.learning_rate,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,            # ★ cfg 노출
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,        # ★ cfg 노출 (sparse reward용 0.97)
            ent_coef=self.cfg.ent_coef,
            vf_coef=self.cfg.vf_coef,
            clip_range=self.cfg.clip_range,
            tensorboard_log=str(self.log_dir),
            device=self.cfg.device,
            seed=self.cfg.seed,
            verbose=1,
            policy_kwargs=dict(
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(item_encoder="deepset"),
                    net_arch=dict(
                        # FT backbone 하이퍼 (굵게!)
                        ft_n_tokens=8,
                        ft_d_model=256,
                        ft_n_heads=8,
                        ft_depth=6,
                        ft_mlp_ratio=6.0,
                        ft_p_drop=0.1,
                        ft_drop_path=0.1,    # ← 위에서 받게 했으면 여기도
                        ft_out_pi=256,
                        ft_out_vf=256,
                        ft_use_rope=True,    # ← 옵션화 했다면
                    ),
                optimizer_kwargs=dict(weight_decay=1e-5)

                ),
        )

        # 3) Writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self._eval_call_idx = 0
        new_logger = configure(str(self.log_dir), ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)

        # 4) ── 플랜 메이커를 '먼저' 만든다 ──
        time_based_plan_maker = make_time_based_plan_maker(self.cfg, self.history)

        # (선택) 어댑티브 컨트롤러와 하이브리드 plan_maker
        self.adaptive_ctl = None
        _override_phase = None

        def plan_maker_hybrid(rollout_idx: int, n_envs: int, *, current_steps: int) -> List[EnvPlan]:
            if _override_phase is None:
                # 적응 전에는 그냥 time_based 사용
                return time_based_plan_maker(rollout_idx, n_envs, current_steps=current_steps)

            phase = int(_override_phase)
            p_fixed = [0.70, 0.50, 0.40][phase]
            rehearsal_p = [
                self.cfg.rehearsal_p_boot,
                self.cfg.rehearsal_p_mid,
                self.cfg.rehearsal_p_late,
            ][phase]

            recorded_pool, folder_weights = _resolve_recorded_pool_and_weights(self.cfg)
            recorded_files = _expand_recorded_pool_to_files(recorded_pool)
            recorded_file_weights = _distribute_folder_weights_to_files(recorded_pool, folder_weights)

            return _make_plans_core(
                seed=self.cfg.seed,
                n_envs=n_envs,
                p_scenario_fixed=p_fixed,
                p_type_sampled=self.cfg.type_sampled_ratio,
                p_synthetic=self.cfg.synthetic_ratio,
                rehearsal_p=rehearsal_p,
                rollout_idx=rollout_idx,
                bin_key="experiment_RL",
                margin_range=(0, 8),
                recorded_file_candidates=recorded_files,
                recorded_file_weights=recorded_file_weights,
                history=self.history,
                selectable_len_choices=self.cfg.selectable_len_choices,
                preview_k_choices=self.cfg.preview_k_choices,
            )

        def inject_adaptive_to_plan_maker(phase_idx: int, p_fixed: float, rehearsal_p: float):
            nonlocal _override_phase
            _override_phase = int(max(_override_phase or 0, phase_idx))

        # ★ 여기서 최종 plan_maker를 '지금' 할당
        self.plan_maker = plan_maker_hybrid if self.cfg.use_adaptive_phase else time_based_plan_maker
        self.eval_plan_maker = make_eval_plan_maker(self.cfg)

        # 5) 콜백 스택을 만든다 (plan_maker를 사용)
        self.callbacks = self._make_callbacks(
            plan_maker=self.plan_maker,
            inject_adaptive_to_plan_maker=inject_adaptive_to_plan_maker if self.cfg.use_adaptive_phase else None
        )

        # 6) ★ 선주입: learn() 전에 첫 reset용 플랜을 미리 넣기
        initial_plans = self.plan_maker(
            0,
            int(getattr(self.env, "num_envs", self.cfg.n_envs_train)),
            current_steps=0,
        )
        for i, p in enumerate(initial_plans):
            payload = env_plan_to_payload(p)
            self.env.env_method("apply_plan", payload, indices=[i])

        # 7) 콜백의 rollout_idx를 1로 시작(중복 방지)
        for cb in self.callbacks.callbacks:
            if isinstance(cb, TrainsetSchedulingCallback):
                cb.rollout_idx = 1
                
    def loadModel(self, path: str, history_dir: Optional[str] = None):
        """
        path: 예) 'planning/RL/PalletFit_RL/logs/MaskablePPO_20251113-174205/ppo_ckpt_24576_steps.zip'
        """
        print(f"[MaskablePPOAgent] Loading pretrained model from: {path}")

        custom_objects = {
            "learning_rate": self.cfg.learning_rate,
            "clip_range": self.cfg.clip_range, 
        }

        # 1) 기존 self.model은 버리고, checkpoint에서 다시 로드
        self.model = MaskablePPO.load(
            path,
            env=self.env,              # 현재 env에 붙여줌
            device=self.cfg.device,    # cuda / cpu
            custom_objects=custom_objects, # 수정된 dict 전달
        )
        
        if history_dir is not None:
            history_log_dir = Path(history_dir)
        else:
            history_log_dir = Path(path).resolve().parent

        self.load_history_from_dir(str(history_log_dir))

        # 2) logger는 새 run 디렉토리로 다시 붙여주기
        new_logger = configure(str(self.log_dir), ["stdout", "csv", "tensorboard"])
        self.model.set_logger(new_logger)

        print("[MaskablePPOAgent] Model successfully loaded and bound to current env.")

    def load_history_from_dir(self, log_dir: str):
        log_dir = Path(log_dir)
        csv_path = log_dir / "dataset_history.csv"

        if csv_path.is_file():
            self.history.load_csv(str(csv_path))
        else:
            print(f"[Agent] No dataset_history.csv found in: {csv_path}")
        
    # 콜백 구성
    def _eval_interval_steps(self) -> int:
        return max(1, int(self.cfg.n_steps) * int(self.cfg.n_envs_train) * int(self.cfg.eval_every_rollouts))

    def _save_interval_steps(self) -> int:
        return max(1, int(self.cfg.n_steps) * int(self.cfg.n_envs_train) * int(self.cfg.save_every_rollouts))

    def _make_callbacks(self, plan_maker, inject_adaptive_to_plan_maker) -> CallbackList:
        cbs: List[BaseCallback] = []

        # Trainset 스케줄링
        cbs.append(TrainsetSchedulingCallback(plan_maker, verbose=1))

        # ★ 데이터셋 히스토리 수집
        cbs.append(HistoryCollectorCallback(self.history, verbose=0))
        # ★ 히스토리 저장(주기 조절 가능)
        cbs.append(HistorySaverCallback(self.history, out_dir=str(self.log_dir), save_every_steps=self._save_interval_steps(), verbose=1))

        # (선택) 평가 + 어댑티브
        adaptive_ctl = None
        if self.cfg.use_adaptive_phase:
            adaptive_ctl = AdaptivePhaseController(
                W=self.cfg.perf_window,
                eps_slope=self.cfg.perf_eps_slope,
                eps_std=self.cfg.perf_eps_std,
                min_eval_gap=self.cfg.perf_min_eval_gap,
            )
        # 체크포인트 콜백 유지
        ckpt = CheckpointCallback(
            save_freq=1,                      # 콜백 내부 카운터용: 1로 둠
            save_path=str(self.log_dir),
            name_prefix="ppo_ckpt",
        )
        wrapped_ckpt = EveryNTimesteps(
            n_steps=self._save_interval_steps(),  # ← 1024 timesteps마다
            callback=ckpt
        )
        cbs.append(wrapped_ckpt)

        cbs.append(EvalAndAdaptiveCallback(
            eval_fn=self.evaluation,
            eval_interval_steps=self._eval_interval_steps(),
            adaptive_ctl=adaptive_ctl,
            inject_adaptive_to_plan_maker=inject_adaptive_to_plan_maker,
            best_model_save_path=str(self.log_dir),
            verbose=1,
        ))
        # cbs.append(AdaptiveHyperparamCallback(
        #     eval_fn=lambda: self.evaluation(episodes=5),
        #     check_freq=self._eval_interval_steps(),
        #     log_dir=str(self.log_dir),  # <--- 여기 추가!
        #     verbose=1
        # ))
        return CallbackList(cbs)

    # ────────────────────────────────────────────────
    # 간단 평가 루틴
    # ────────────────────────────────────────────────
    @th.no_grad()
    def evaluation(
        self,
        episodes: Optional[int] = None,
        max_eval_steps: int = 200,
    ) -> Dict[str, float]:
        """
        SubprocVecEnv 병렬 평가. 0번 워커는 자동으로 GIF를 저장한다(env 내부 처리).

        Args:
            episodes: 평가 에피소드 수. None이면 n_envs_eval(한 라운드).
            max_eval_steps: 에피소드당 step 상한(VecEnv 한 라운드 기준 안전장치).
        """
        self.model.policy.set_training_mode(False)

        n_envs = int(self.eval_env.num_envs)
        if episodes is None:
            episodes = n_envs

        # 평가 plan 캐싱(재현성). 라운드별로 라운드-사이즈만큼 잘라서 사용.
        if getattr(self, "_cached_eval_plans", None) is None:
            print(f"Generating and caching static eval plans ({episodes} episodes)...")
            plan_maker = make_eval_plan_maker(self.cfg)
            self._cached_eval_plans = plan_maker(0, episodes, current_steps=0)
        plans: List[EnvPlan] = self._cached_eval_plans or []

        # plan 길이 부족 시 마지막 plan 반복
        if len(plans) < episodes and plans:
            plans = list(plans) + [plans[-1]] * (episodes - len(plans))

        # NOOP 인덱스(라운드에서 이미 끝난 워커에 보낼 더미 액션)
        try:
            noop_idx = int(self.eval_env.get_attr("NOOP_IDX", indices=[0])[0])
        except Exception:
            noop_idx = int(PREVIEW_MAX * ACTION_MAX_CANDIDATES) - 1

        su_list: List[float] = []
        packed_list: List[int] = []

        plan_idx = 0
        round_idx = 0
        while plan_idx < episodes:
            round_size = min(n_envs, episodes - plan_idx)

            # 이번 라운드 plan을 워커에 주입
            for i in range(round_size):
                payload = env_plan_to_payload(plans[plan_idx + i])
                self.eval_env.env_method("apply_plan", payload, indices=[i])

            # pending_plan 소비를 위해 강제 reset
            obs = self.eval_env.reset()
            done_mask = np.zeros(n_envs, dtype=bool)
            done_mask[round_size:] = True  # 미사용 워커는 처음부터 done 취급

            for _ in range(max_eval_steps):
                if done_mask.all():
                    break
                masks = get_action_masks(self.eval_env)
                actions, _ = self.model.predict(
                    obs, deterministic=True, action_masks=masks
                )
                actions = np.asarray(actions, dtype=np.int64)
                # 이미 끝난 워커는 NOOP으로 패딩(자동 재리셋되더라도 GIF는 안 찍힘)
                if done_mask.any():
                    actions[done_mask] = noop_idx
                obs, _, dones, infos = self.eval_env.step(actions)
                for i in range(round_size):
                    if dones[i] and not done_mask[i]:
                        done_mask[i] = True
                        info = infos[i] if isinstance(infos[i], dict) else {}
                        su_list.append(float(info.get("SU", 0.0)))
                        packed_list.append(int(info.get("packed_count", 0)))

            # 미완료 워커는 실패로 간주 → 0.0 기록 (라운드 정합성)
            for i in range(round_size):
                if not done_mask[i]:
                    print(f"[Eval] env{i} did not finish within {max_eval_steps} steps")
                    su_list.append(0.0)
                    packed_list.append(0)

            plan_idx += round_size
            round_idx += 1

        mean_su = float(np.mean(su_list)) if su_list else 0.0
        mean_packed = float(np.mean(packed_list)) if packed_list else 0.0

        self.model.policy.set_training_mode(True)
        return {"SU": mean_su, "packed": mean_packed}

    def train(self):
        # --- 0) log_dir 준비 & 소스 파일 스냅샷 ---
        os.makedirs(self.log_dir, exist_ok=True)

        src_dir = os.path.join("planning", "RL", "PalletFit_RL")
        snapshot_files = [
            "agent.py",
            "config.py",
            "env.py",
            "obs_builder.py",
            "reward_builder.py",
            "act_builder.py",
            "custom_value_policy.py",
        ]
        for fname in snapshot_files:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(self.log_dir, fname)
            try:
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"[train] skip snapshot: not found -> {src}")
            except Exception as ce:
                print(f"[train] snapshot copy failed for {fname}: {ce}")

        try:
            self.model.learn(
                total_timesteps=int(self.cfg.total_timesteps),
                callback=self.callbacks,
                progress_bar=True
            )

            # --- 1) 학습 완료 후 최종 모델 저장 ---
            final_model_path = self.log_dir / "final_model.zip"
            self.model.save(str(final_model_path))
            print(f"[train] Final model saved to: {final_model_path}")

        finally:
            self.writer.flush()
            self.writer.close()



# =============================================================================
# 7) main
# =============================================================================

# def smoke_test_env():
#     print("🚬 [Smoke Test] 환경 무결성 검사 시작...")
    
#     # 1. 환경 초기화
#     # (Config에서 사용하는 경로가 있다면 맞춰주세요)
#     env = PalletFitEnv()
    
#     # 테스트용 플랜 설정 (오프라인 파일 하나 지정)
#     test_plan = {
#         "item_mode": "recorded",
#         "item_payload": {
#             "recorded_paths": ["planning/data/Item_data/paper/setting123_discrete/dataset_episode_000.json"],
#             "episode_seed": 123
#         },
#         "bin": "experiment_RL"
#     }
#     env.apply_plan(test_plan)
    
#     obs, info = env.reset()
    
#     done = False
#     step_count = 0
#     total_reward = 0
    
#     while not done:
#         step_count += 1
        
#         # 2. 유효한 액션 마스킹 확인
#         valid_actions = np.where(obs['act_mask'] == 1)[0]
        
#         # NOOP(아무것도 안 함) 제외하고 실제 적재 액션이 있는지 확인
#         real_actions = [a for a in valid_actions if a != env.NOOP_IDX]
        
#         if len(real_actions) > 0:
#             # 랜덤으로 적재 액션 선택
#             action = np.random.choice(real_actions)
#             action_type = "PLACE"
#         else:
#             # 적재할 곳이 없으면 NOOP (종료 가능성 높음)
#             action = env.NOOP_IDX
#             action_type = "NOOP"
            
#         # 3. Step 실행
#         obs, reward, terminated, truncated, info = env.step(action)
#         total_reward += reward
#         done = terminated or truncated
        
#         # 4. 상태 확인
#         current_bin_size = env.packer.current_bin.size
#         print(f"   Step {step_count}: Action={action_type}({action}) -> Reward={reward:.4f}, BinSize={current_bin_size}}")
        
#         # [검증] 적재 액션을 했는데 Bin Size가 안 늘어나면 문제
#         if action_type == "PLACE" and reward > 0:
#             if current_bin_size < step_count: 
#                 print(f"❌ [Critical] 적재 성공했는데 Bin Size가 증가하지 않음! (Current: {current_bin_size})")
#                 return

#     print(f"🏁 에피소드 종료. 총 스텝: {step_count}, 최종 Bin Size: {env.packer.current_bin.size}, SU: {info['SU']:.4f}")
    
#     if env.packer.current_bin.size > 0:
#         print("✅ [Pass] 아이템이 정상적으로 적재되었습니다.")
#     else:
#         print("❌ [Fail] 아이템이 하나도 적재되지 않았습니다.")

# if __name__ == "__main__":
#     smoke_test_env()


# if __name__ == "__main__":
#     import torch
    
#     torch.cuda.empty_cache()

#     # old_log_dir = "planning/RL/PalletFit_RL/logs/MaskablePPO_20251222-124925"

#     agent = MaskablePPOAgent(cfg=AgentConfig())
#     agent.loadModel(
#         'planning/RL/PalletFit_RL/logs/MaskablePPO_20251222-124925/ppo_ckpt_6500376_steps.zip'
#     )
#     # 방법 1) loadModel에 history_dir 인자로 넘기기
#     # agent.loadModel(
#     #     f'{old_log_dir}/final_model.zip',
#     #     history_dir=old_log_dir,
#     # )
#     # agent.loadModel("planning/RL/PalletFit_RL/logs/MaskablePPO_20251128-110723/ppo_ckpt_2621440_steps.zip")
#     agent.train()



# =============================================================================
# 무한 학습 코드
# =============================================================================


if __name__ == "__main__":
    import torch
    import glob
    import re
    import argparse

    torch.cuda.empty_cache()

    # =========================================================================
    # ★ CLI 옵션 — preset 선택
    # =========================================================================
    parser = argparse.ArgumentParser(description="PalletFit-RL training entry")
    parser.add_argument(
        "--preset", choices=["default", "lite"], default="default",
        help="default = 16C16T + A6000 48GB 기준 / lite = 4C8T + 6GB GPU 호환 (논문 재현용)",
    )
    args, _ = parser.parse_known_args()

    # =========================================================================
    # ★ [설정] 경로 지정
    # =========================================================================
    LOG_DIR_ROOT = "planning/RL/PalletFit_RL/logs/MaskablePPO_AutoResume"
    # 기존에 학습하던 모델이 있다면 여기에 지정 (없으면 None)
    SPECIFIC_START_MODEL = "planning/RL/PalletFit_RL/logs/MaskablePPO_20251212-122844/ppo_ckpt_20512_steps.zip"
    # =========================================================================

    # 1. 에이전트 생성 시 경로 주입 (이제 모든 Callback이 이 경로를 봅니다)
    cfg = make_lite_config() if args.preset == "lite" else AgentConfig()
    print(f"⚙️  Preset: {args.preset}  (n_envs_train={cfg.n_envs_train}, "
          f"n_steps={cfg.n_steps}, batch_size={cfg.batch_size})")
    agent = MaskablePPOAgent(cfg=cfg, override_log_dir=LOG_DIR_ROOT)
    
    # Logger 다시 세팅 (확실하게 하기 위해)
    new_logger = configure(str(agent.log_dir), ["stdout", "csv", "tensorboard"])
    agent.model.set_logger(new_logger)

    # 2. 자동 이어하기 로직
    # 우선 AutoResume 폴더 안에 있는 가장 최신 ckpt 찾기
    new_checkpoints = glob.glob(os.path.join(LOG_DIR_ROOT, "*.zip"))
    
    target_ckpt = None
    target_history_dir = None

    if new_checkpoints:
        # A. AutoResume 폴더에 파일이 있음 -> 무한 루프가 돌고 있다는 뜻
        #    가장 최신 모델을 로드하고, 히스토리도 AutoResume 폴더에서 읽음
        target_ckpt = max(new_checkpoints, key=os.path.getctime)
        target_history_dir = LOG_DIR_ROOT  # ★ 여기가 중요! 자기 자신 폴더를 봄
        print(f"🔄 [Auto-Resume] Found ongoing checkpoint: {target_ckpt}")

    elif os.path.isfile(SPECIFIC_START_MODEL):
        # B. AutoResume은 비어있지만, 사용자가 지정한 시작 파일이 있음 (첫 이사)
        target_ckpt = SPECIFIC_START_MODEL
        # 히스토리는 '이사 오기 전' 폴더에서 가져옴
        target_history_dir = str(Path(SPECIFIC_START_MODEL).parent)
        print(f"🔄 [Start] Importing model from: {target_ckpt}")
        print(f"🔄 [Start] Importing history from: {target_history_dir}")
        
        # (선택) 첫 이사 때만 히스토리 파일 복사해오기 (안전빵)
        try:
            old_csv = Path(target_history_dir) / "dataset_history.csv"
            new_csv = Path(LOG_DIR_ROOT) / "dataset_history.csv"
            if old_csv.exists() and not new_csv.exists():
                shutil.copy2(old_csv, new_csv)
                print("📦 Copied history CSV to new home.")
        except Exception as e:
            print(f"⚠️ History copy failed: {e}")

    else:
        # C. 아무것도 없음 -> 쌩 처음
        print("🆕 [Start] No checkpoint found. Starting fresh training.")

    # 3. 모델 & 히스토리 로드
    if target_ckpt:
        try:
            # history_dir를 명시적으로 넘김
            agent.loadModel(target_ckpt, history_dir=target_history_dir)
            
            # Timesteps 복구
            match = re.search(r"(\d+)_steps", str(target_ckpt))
            if not match: 
                match = re.search(r"ckpt_(\d+)", str(target_ckpt))
            
            if match:
                prev_steps = int(match.group(1))
                agent.model.num_timesteps = prev_steps
                print(f"✅ Set model.num_timesteps to {prev_steps}")

                # Rollout Index 복구
                steps_per_rollout = cfg.n_envs_train * cfg.n_steps
                restored_rollout_idx = prev_steps // steps_per_rollout
                
                for cb in agent.callbacks.callbacks:
                    if isinstance(cb, TrainsetSchedulingCallback):
                        cb.rollout_idx = restored_rollout_idx + 1
                        print(f"✅ Restored rollout_idx to {cb.rollout_idx}")
                        break

        except Exception as e:
            print(f"❌ Final load failed: {e}")
            exit(1)

    # 4. 학습 실행
    CHUNK_SIZE = cfg.n_envs_train * cfg.n_steps * 4 # 원하는 만큼 조절
    print(f"🚀 Training chunk for {CHUNK_SIZE} steps...")

    agent.model.learn(
        total_timesteps=CHUNK_SIZE,
        callback=agent.callbacks,
        reset_num_timesteps=False, 
        progress_bar=True
    )
    
    # 5. 저장 (AutoResume 폴더에 저장됨)
    save_path = agent.log_dir / f"ppo_ckpt_{agent.model.num_timesteps}_steps.zip"
    agent.model.save(str(save_path))
    
    # 히스토리 강제 저장 (HistorySaverCallback이 해주지만 안전하게 한 번 더)
    agent.history.save_csv(str(agent.log_dir / "dataset_history.csv"))
    
    print(f"✅ Chunk finished. Saved to {save_path}. Exiting...")