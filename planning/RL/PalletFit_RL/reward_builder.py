# planning/RL/PalletFit_RL/reward_builder.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

# utils.overlap에 compute_overlap_area가 있다고 가정 (없으면 아래 내부 함수 사용)
from utils.overlap import compute_overlap_area 
from utils.get_value import (
    balance_term_capped,
    get_direction_overlap,
    collect_ez_stats
)

# ─────────────────────────────────────────────
# 1. 하이퍼파라미터 설정 (효율성 우선 + 안정성 보너스)
# ─────────────────────────────────────────────
ALIVE_BONUS: float = 0.0  # 적재 성공 시 기본 지급

# [1순위: 효율성 (Efficiency)] - 점수 엔진
SU_WEIGHT: float = 1.6           # 채움 보상 (가장 큼)
DEAD_WEIGHT: float = 1.0         # 낭비 페널티 (큼)

# [2순위: 안정성 (Stability)] - 가이드라인 (Soft Constraint)
# 이미 Action Masking에서 최소 조건(예: 0.6)은 통과했음.
# 여기서는 "얼마나 더 완벽하게(1.0) 지지받는가"에 대한 가중치.
# (1.0 - support_ratio) 만큼 감점하는 방식 사용 추천.
STABILITY_PENALTY_WEIGHT: float = 0.5 

# [3순위: 품질 (Quality)] - 보너스
FLATNESS_WEIGHT: float = 0.002
CONTACT_WEIGHT: float = 0.1
BALANCE_WEIGHT: float = 0.5

# checkPivot의 에러 코드 상수들을 가져오거나 직접 정의 (매핑용)
# utils/checkPivot.py의 값과 일치해야 함
FAIL_OUT_OF_BOUNDS_NEG = -1
FAIL_OUT_OF_BOUNDS_POS = -2
FAIL_COLLISION = -3
FAIL_WEIGHT_EXCEEDED = -4
FAIL_NO_TOP_EMPTY = -5
FAIL_NO_SUPPORT_BOTTOM = -6
FAIL_SUPPORT_OVERLOAD = -7
FAIL_OVERHANG_TOO_MUCH = -8
FAIL_SUPPORT_AREA_INSUFFICIENT = -9
FAIL_CG_OUTSIDE_SUPPORT = -10
FAIL_CUMULATIVE_UNSTABLE = -11

# ─────────────────────────────────────────────
# [설정] 실패 원인별 벌점 (Penalty Map)
# ─────────────────────────────────────────────
PENALTY_MAP = {
    # 1. 치명적 물리 오류 (충돌, 경계 이탈) -> 매우 큰 벌점
    FAIL_OUT_OF_BOUNDS_NEG: -5.0,
    FAIL_OUT_OF_BOUNDS_POS: -5.0,
    FAIL_COLLISION:         -5.0,
    FAIL_NO_TOP_EMPTY:      -5.0,

    # 2. 불안정성/지지력 부족 -> 중간 벌점
    FAIL_NO_SUPPORT_BOTTOM: -3.0,  # 공중 부양
    FAIL_CG_OUTSIDE_SUPPORT: -3.0, # 무게중심 이탈
    FAIL_CUMULATIVE_UNSTABLE: -3.0,

    # 3. 기타 제약 위반 -> 기본 벌점
    FAIL_WEIGHT_EXCEEDED:   -3.0,
    FAIL_SUPPORT_OVERLOAD:  -3.0,
    FAIL_OVERHANG_TOO_MUCH: -2.0,
    FAIL_SUPPORT_AREA_INSUFFICIENT: -2.0,
    
    # 시스템적 에러 (Masked Action 선택 등)
    "default": -5.0 
}
# ─────────────────────────────────────────────
# 2. 보조 계산 함수들
# ─────────────────────────────────────────────

def _dead_ratio(bin_obj) -> float:
    try:
        r = float(bin_obj.get_deadVolume_ratio())
        if not (r == r): return 0.0
        return max(0.0, min(1.0, r))
    except Exception:
        return 0.0

def _calculate_support_ratio_geometric(bin_obj, item) -> float:
    """
    Bin의 R-tree 기반 공간 탐색(get_bottom_items)을 사용하여
    현재 아이템의 바닥 지지율을 계산합니다.
    """
    # 1. 아이템 바닥 면적 계산
    w, h, _ = item.getDimension() # (x, y, z)
    item_area = float(w) * float(h)
    if item_area < 1e-6:
        return 0.0

    # 2. 바닥(z=0)에 붙어있는 경우 100% 지지
    if item.b_position[2] <= 1e-3:
        return 1.0

    # 3. 바로 아래(z)에 있는 아이템들을 R-tree로 탐색
    # bin.py에 구현된 get_bottom_items는 loaded_item의 vertices를 기반으로
    # search_xyz를 수행하므로 후보군 탐색에 적합합니다.
    bottom_candidates = bin_obj.get_bottom_items(item)
    
    if not bottom_candidates:
        return 0.0

    supported_area = 0.0
    
    # 4. 겹치는 면적 계산 (Topology가 아닌 Geometry 기반)
    # item의 바닥면 정보
    _, item_bottom_bounds = item.getFaceInfo('bottom') 
    # bounds format: {'x': (min, max), 'y': (min, max), 'z': ...}

    for b_item in bottom_candidates:
        if b_item is None: continue
        
        # 아래 물체의 윗면 정보
        _, b_top_bounds = b_item.getFaceInfo('top')
        
        # overlap_intervals 로직을 사용하여 면적 계산
        # x축 겹침 길이
        x_overlap = max(0.0, min(item_bottom_bounds['x'][1], b_top_bounds['x'][1]) - 
                             max(item_bottom_bounds['x'][0], b_top_bounds['x'][0]))
        # y축 겹침 길이
        y_overlap = max(0.0, min(item_bottom_bounds['y'][1], b_top_bounds['y'][1]) - 
                             max(item_bottom_bounds['y'][0], b_top_bounds['y'][0]))
        
        supported_area += (x_overlap * y_overlap)

    # 비율 반환 (최대 1.0)
    return min(1.0, supported_area / item_area)

def _get_surface_std(bin_obj) -> float:
    counts, _ = collect_ez_stats(bin_obj)
    if not counts: return 0.0
    heights = []
    for h, count in counts.items():
        heights.extend([h] * count)
    if not heights: return 0.0
    return float(np.std(heights))

def _get_contact_score(bin_obj, item) -> float:
    try:
        overlaps = get_direction_overlap(item, bin_obj, palletizing_mode=True)
        return sum([v[3] for v in overlaps.values()])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# 3. 메인 리워드 함수 (build_reward)
# ─────────────────────────────────────────────

def build_reward(
    *,
    before_bin,
    after_bin,
    cand_item=None,
    terminated: bool,
    finished: bool,
    truncated: bool = False,
    failure_code = None  # [추가] 실패 코드 받기
) -> Tuple[float, Dict[str, float]]:
    
    # ── 0) 실패 종료 처리 (Failure Handling) ───────────────────────
    if terminated and not finished:
        # failure_code에 따른 벌점 조회
        penalty = PENALTY_MAP.get("default")
        
        if failure_code is not None:
            if failure_code in PENALTY_MAP:
                penalty = PENALTY_MAP[failure_code]
            elif isinstance(failure_code, str):
                # 문자열 에러(시스템 에러)인 경우
                penalty = -0.005
        
        terms = {k: 0.0 for k in ["eff_su", "eff_dead", "stab_soft", "qual_flat", "qual_contact", "bal"]}
        # 가능한 경우의 수만 제공할때는 penalty역할 거의 없애기
        penalty *= 0.001
        terms["fail"] = penalty
        
        # (옵션) 로그에 어떤 에러로 죽었는지 기록하고 싶다면 terms에 추가할 수도 있음
        # terms["fail_type"] = float(failure_code) if isinstance(failure_code, int) else 0.0
        
        return float(penalty), terms

    # ── 1) [Efficiency] 공간 효율성 (SU & Dead Volume) ────────
    su_gain = (float(after_bin.SU) - float(before_bin.SU)) * 10.0
    r_su = SU_WEIGHT * su_gain

    dead_increase = _dead_ratio(after_bin) - _dead_ratio(before_bin)
    r_dead = 0.0
    if dead_increase > 1e-4:
        r_dead = -1.0 * DEAD_WEIGHT * (dead_increase * 10.0)

    # ── 2) [Stability] 안정성 소프트 제약 (Soft Constraint) ──
    # Action Masking에 의해 이미 최소 조건은 통과했습니다.
    # 여기서는 100% 지지에 가까울수록 0에 수렴하고, 
    # 최소 조건에 가까울수록(불안정할수록) 음의 점수를 주는 방식을 사용합니다.
    # 예: 지지율 1.0 -> 0점 (완벽)
    #     지지율 0.6 -> (0.6 - 1.0) * 0.5 = -0.2점 (감점)
    
    r_stability = 0.0
    if cand_item is not None:
        # 새로 구현한 Geometric 기반 지지율 계산 함수 사용
        support_ratio = _calculate_support_ratio_geometric(after_bin, cand_item)
        
        # Soft Penalty 방식: 1.0(완벽)에서 부족한 만큼 감점
        # 효율성(SU)이 매우 크다면 이 감점을 감수하고 적재할 것입니다.
        r_stability = (support_ratio - 1.0) * STABILITY_PENALTY_WEIGHT

    # ── 3) [Quality] 적재 품질 (Bonus) ────────────────────────
    # std_before = _get_surface_std(before_bin)
    # std_after = _get_surface_std(after_bin)
    # flatness_gain = std_before - std_after 
    # r_flat = FLATNESS_WEIGHT * flatness_gain

    # Contact Score Scaling (면적 단위라 값이 큼 -> 스케일링 필요)
    # 아이템 바닥 면적으로 나누어 비율로 변환하거나 상수로 나눔
    contact_val = _get_contact_score(after_bin, cand_item) if cand_item else 0.0
    # 대략적인 스케일링 (mm^2 단위라고 가정 시 100x100=10000)
    r_contact = CONTACT_WEIGHT * (contact_val / 10000.0) 

    # ── 4) [Balance] 무게 중심 ────────────────────────────────
    bal_term_val = balance_term_capped(r_su, after_bin, cand_item=None)
    r_bal = BALANCE_WEIGHT * bal_term_val

    # ── 5) 최종 합산 ──────────────────────────────────────────
    total_reward = (
        ALIVE_BONUS 
        + r_su 
        + r_dead 
        + r_stability 
        # + r_flat 
        + r_contact 
        + r_bal
    )

    terms: Dict[str, float] = {
        "eff_su": float(r_su),
        "eff_dead": float(r_dead),
        "stab_soft": float(r_stability),
        # "qual_flat": float(r_flat),
        "qual_contact": float(r_contact),
        "bal": float(r_bal),
    }

    return float(total_reward), terms