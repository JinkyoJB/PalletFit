# planning/RL/PalletFit_RL/reward_builder.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

# from utils.overlap import compute_overlap_area 
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
    FAIL_COLLISION: -5.0,
    FAIL_NO_TOP_EMPTY: -5.0,

    # 2. 불안정성/지지력 부족 -> 중간 벌점
    FAIL_NO_SUPPORT_BOTTOM: -3.0,  # 공중 부양
    FAIL_CG_OUTSIDE_SUPPORT: -3.0,  # 무게중심 이탈
    FAIL_CUMULATIVE_UNSTABLE: -3.0,

    # 3. 기타 제약 위반 -> 기본 벌점
    FAIL_WEIGHT_EXCEEDED: -3.0,
    FAIL_SUPPORT_OVERLOAD: -3.0,
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
        if not (r == r):
            return 0.0
        return max(0.0, min(1.0, r))
    except Exception:
        return 0.0


def _calculate_support_ratio_geometric(bin_obj, item) -> float:
    """
    Bin의 R-tree 기반 공간 탐색(get_bottom_items)을 사용하여
    현재 아이템의 바닥 지지율을 계산합니다.
    """
    # 1. 아이템 바닥 면적 계산
    w, h, _ = item.getDimension()   # (x, y, z)
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
        if b_item is None: 
            continue
        
        # 아래 물체의 윗면 정보
        _, b_top_bounds = b_item.getFaceInfo('top')
        
        # overlap_intervals 로직을 사용하여 면적 계산
        # x축 겹침 길이
        x_overlap = max(0.0, min(item_bottom_bounds['x'][1], b_top_bounds['x'][1]) - max(item_bottom_bounds['x'][0], b_top_bounds['x'][0]))
        # y축 겹침 길이
        y_overlap = max(0.0, min(item_bottom_bounds['y'][1], b_top_bounds['y'][1]) - max(item_bottom_bounds['y'][0], b_top_bounds['y'][0]))
        
        supported_area += (x_overlap * y_overlap)

    # 비율 반환 (최대 1.0)
    return min(1.0, supported_area / item_area)


def _get_surface_std(bin_obj) -> float:
    counts, _ = collect_ez_stats(bin_obj)
    if not counts:
        return 0.0
    heights = []
    for h, count in counts.items():
        heights.extend([h] * count)
    if not heights:
        return 0.0
    return float(np.std(heights))


def _get_contact_score(bin_obj, item) -> float:
    try:
        overlaps = get_direction_overlap(item, bin_obj, palletizing_mode=True)
        return sum([v[3] for v in overlaps.values()])
    except Exception:
        return 0.0

# ─────────────────────────────────────────────
# 3. 메인 점수 함수 (build_reward)
# ─────────────────────────────────────────────
#   build_reward(bin_obj, placed_item=None) - 현재 bin의 절대 점수를 반환
#   env가 step n과 step n+1의 점수 차이로 reward를 계산.
#   deepcopy 비용 제거 + state-based potential shaping과 동일한 효과.
# ─────────────────────────────────────────────


_TERM_KEYS = ("eff_su", "eff_dead", "bal", "alive", "stab_soft", "qual_contact")


def build_reward(bin_obj, placed_item=None) -> Tuple[float, Dict[str, float]]:
    """
    현재 bin 상태의 절대 점수를 반환.

    env에서 step n의 점수(`prev`)와 step n+1의 점수(`curr`)를 각각 계산한 뒤
    `reward = curr - prev`로 사용한다.

    - State-based 항목(r_su, r_dead, r_bal): bin 상태에만 의존 → delta가 자연스럽게 계산됨.
    - Placement-specific 항목(r_alive, r_stability, r_contact): placed_item이 있을 때만
      적용. env는 step 후 prev_score에서 이 항목들을 빼서 다음 step용 baseline을 갱신한다.

    Args:
        bin_obj: 현재 bin (after-step 또는 reset 상태).
        placed_item: 방금 놓인 아이템. None이면 placement bonus 항목 0.

    Returns:
        total_score: 항목 합계 (절대 점수).
        terms: 항목별 분해 dict — eff_su, eff_dead, bal, alive, stab_soft, qual_contact.
    """
    # ── 1) State-based terms (bin 상태에만 의존) ─────────────
    r_su = SU_WEIGHT * float(bin_obj.SU) * 10.0
    r_dead = -1.0 * DEAD_WEIGHT * _dead_ratio(bin_obj) * 10.0

    # Balance: state-based 전환에 따라 cap을 고정 reference(1.0)로 사용.
    # (이전 design은 r_su delta에 비례한 cap이라 매 step의 balance 영향이 SU 증분에 묶여 있었음.
    #  state-based로 바뀌면서 cap을 절대 SU에 비례시키면 후반 step에서 balance가 폭주하므로
    #  고정 reference로 안정화. 필요 시 BALANCE_WEIGHT로 강도 조절.)
    try:
        bal_term_val = balance_term_capped(1.0, bin_obj, cand_item=None)
    except Exception:
        bal_term_val = 0.0
    r_bal = BALANCE_WEIGHT * bal_term_val

    # ── 2) Placement-specific bonus (placed_item에만 의존) ──
    r_alive = 0.0
    r_stability = 0.0
    r_contact = 0.0

    if placed_item is not None:
        r_alive = ALIVE_BONUS

        try:
            support_ratio = _calculate_support_ratio_geometric(bin_obj, placed_item)
            r_stability = (support_ratio - 1.0) * STABILITY_PENALTY_WEIGHT
        except Exception:
            r_stability = 0.0

        try:
            contact_val = _get_contact_score(bin_obj, placed_item)
            r_contact = CONTACT_WEIGHT * (contact_val / 10000.0)
        except Exception:
            r_contact = 0.0

    # ── 3) 합산 ──────────────────────────────────────────────
    total_score = r_su + r_dead + r_bal + r_alive + r_stability + r_contact

    terms: Dict[str, float] = {
        "eff_su":       float(r_su),
        "eff_dead":     float(r_dead),
        "bal":          float(r_bal),
        "alive":        float(r_alive),
        "stab_soft":    float(r_stability),
        "qual_contact": float(r_contact),
    }
    return float(total_score), terms


def get_failure_penalty(failure_code=None) -> Tuple[float, Dict[str, float]]:
    """
    실패 종료 시 페널티 반환. env가 직접 호출.

    이전 build_reward의 failure 분기를 분리한 것. 정상 step의 state-delta 보상과
    경로가 다르므로 별도 함수로 둔다.
    """
    penalty = PENALTY_MAP.get("default", -5.0)

    if failure_code is not None:
        if failure_code in PENALTY_MAP:
            penalty = PENALTY_MAP[failure_code]
        elif isinstance(failure_code, str):
            # 문자열 에러(NOOP, RETRY_LIMIT 등 시스템 에러)
            penalty = -0.005

    # 가능한 경우의 수만 제공할 때는 penalty 역할 거의 없애기 (기존 정책 유지)
    penalty *= 0.001

    terms: Dict[str, float] = {k: 0.0 for k in _TERM_KEYS}
    terms["fail"] = float(penalty)
    return float(penalty), terms
