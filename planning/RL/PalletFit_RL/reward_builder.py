# planning/RL/PalletFit_RL/reward_builder.py
from __future__ import annotations

from typing import Dict, Tuple

# from utils.overlap import compute_overlap_area
from utils.get_value import (
    balance_feature,
    normalize_balance_features,
    get_direction_overlap,
)

# ─────────────────────────────────────────────
# 1. 하이퍼파라미터 설정 (효율성 우선 + 안정성 보너스)
# ─────────────────────────────────────────────
# [Per-placement 정액 보너스] - dense gradient용
# 매 successful placement마다 동일 값 → 작은-큰 박스 모두 placement 자체에 동일 신호.
# (옛 ALIVE_BONUS의 새 의미: "뭐든 놓으면 +" — 작은 박스 무시 방지)
ALIVE_BONUS: float = 0.5

# [1순위: 효율성 (Efficiency)] - 점수 엔진
# ★ 옵션 D 적용 (2026-05-01): per-step ΔSU 신호를 제거하고
#   에피소드 종료(finished) 시 TERMINAL_SU_WEIGHT × final_SU × 10 한 번에 지급.
#   이유: 큰-작은 박스의 ΔV가 수십~수백 배 차이라서 매 step 분산 시
#   "큰 박스 먼저" 그리디 편향이 강했음. terminal 일괄 지급은 진짜 목표(final SU)에 직접 align.
SU_WEIGHT: float = 0.0           # ★ per-step에서 0. (의도적 disable)
TERMINAL_SU_WEIGHT: float = 5.0  # ★ 에피소드 종료 시 final_SU 보너스 가중치
# ★ DEAD_WEIGHT: 1.0 → 0.3 (2026-05-02, r_dead 컨셉 정리)
#   margin-bridge로 생긴 dead zone은 정의상 필연적이고 동시에 transport 안정성에 기여.
#   이걸 별도 geometry-detection으로 면제하는 대신, dead 페널티를 "약한 가이드"로 약화하고
#   진짜 placement 평가는 r_stability + r_contact가 담당하게 함.
#   (margin-bridge: 지지 면적 큼 → r_stability ↑, 두 아이템과 접촉 → r_contact ↑ → r_dead 자연 상쇄)
DEAD_WEIGHT: float = 0.3

# [2순위: 안정성 (Stability)] - 가이드라인 (Soft Constraint)
# 이미 Action Masking에서 최소 조건(예: 0.6)은 통과했음.
# 여기서는 "얼마나 더 완벽하게(1.0) 지지받는가"에 대한 가중치.
# (1.0 - support_ratio) 만큼 감점하는 방식 사용 추천.
STABILITY_PENALTY_WEIGHT: float = 0.5 

# [3순위: 품질 (Quality)] - 보너스
# (FLATNESS_WEIGHT 제거 — 2026-05-02. r_flat 신호는 obs.h_score와 정보 중복 + 옵션 D 철학 충돌로 영구 삭제. Sec 7-9 참조.)
# ★ CONTACT_WEIGHT: 0.1 → 1.0 (2026-05-02, 스케일 정규화 동시 변경)
#   이전 r_contact = 0.1 × (contact_mm² / 10000)는 상한 없고 박스/bin 크기에 비례 → step당 +1.25~+5.0
#   디버그 트레이스에서 qual_contact 누적이 terminal_su(+50)보다 큰 +60 → 옵션 D의 진짜 목표 신호 압도.
#   변경 후: r_contact = 1.0 × (contact_mm² / 측면적합) ∈ [0, 1.0] per step. 누적 ~15 (alive 16과 비슷).
CONTACT_WEIGHT: float = 1.0

# ★ r_bal: per-step → terminal로 이전 (2026-05-02, 옵션 D 철학 일관성)
#   이전: BALANCE_WEIGHT × balance_term_capped(1.0, ...) per-step → ±0.025/step,
#         32-item 누적 -0.016 (사실상 0%, 학습 신호 무영향).
#   "transport 균형"은 매 step 변화가 아니라 episode 전체 적재의 최종 결과 → terminal로 이전.
#   bal_score ∈ [0, 1] (1=완벽 균형) → terminal_bal = TERMINAL_BAL_WEIGHT × bal_score × 10 ∈ [0, 5].
TERMINAL_BAL_WEIGHT: float = 0.5

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
# 정책적 실패 페널티 (2026-05-02, 옵션 D 후속 수정)
#   env.py가 실제로 전달하는 failure_code는 "NOOP" / "RETRY_LIMIT" 두 가지뿐.
#   이건 agent의 직접 책임(policy 결정)이라 명시적 페널티를 부여한다.
#
#   값 산정 근거:
#   - per-placement alive bonus = 0.5
#   - 정상 완주 ep_return ≈ alive 16 + terminal_su 50 + (contact 60) ≈ 125 (TSG 32-item 기준)
#   - NOOP/RETRY = -0.5는 "1 placement 분의 명시적 손실" + terminal_su 미지급(~50)이라는
#     거대한 기회비용을 합쳐 강한 회피 신호가 됨.
#   - 너무 크게 잡으면 초기 exploration이 NOOP을 한 번이라도 골랐을 때 학습이 출렁임.
# ─────────────────────────────────────────────
STRING_PENALTIES: Dict[str, float] = {
    "NOOP":        -0.5,   # 자발적 종료 — terminal_su 기회비용과 합쳐 강력한 회피
    "RETRY_LIMIT": -0.5,   # 물리 재시도 한계 초과 — 나쁜 placement 반복 선택
}

# 물리적 실패(PENALTY_MAP 정수 키)는 action masking이 거의 다 잡아주므로 가벼운 감쇠 유지.
# 단 기존 × 0.001 → × 0.01로 10× 강화 (혹시 mask가 새는 edge case에 더 명확한 신호).
PENALTY_SCALE_PHYSICAL: float = 0.01
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


def _get_contact_score(bin_obj, item) -> float:
    """
    item의 4개 측면(L/R/F/B) 접촉 면적 합(mm²)을 반환.
    상한: item의 측면적 합 = 2 × D × (W + H).
    """
    try:
        overlaps = get_direction_overlap(item, bin_obj, palletizing_mode=True)
        return sum([v[3] for v in overlaps.values()])
    except Exception:
        return 0.0


def _item_lateral_area(item) -> float:
    """item의 4개 측면 면적 합 (= 측면 접촉의 이론적 최대)."""
    try:
        w, h, d = item.getDimension()
        return 2.0 * float(d) * (float(w) + float(h))
    except Exception:
        return 0.0

# ─────────────────────────────────────────────
# 3. 메인 점수 함수 (build_reward)
# ─────────────────────────────────────────────
#   build_reward(bin_obj, placed_item=None) - 현재 bin의 절대 점수를 반환
#   env가 step n과 step n+1의 점수 차이로 reward를 계산.
#   deepcopy 비용 제거 + state-based potential shaping과 동일한 효과.
# ─────────────────────────────────────────────


_TERM_KEYS = ("eff_su", "eff_dead", "alive", "stab_soft", "qual_contact",
              "terminal_su", "terminal_bal")


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
        terms: 항목별 분해 dict — eff_su, eff_dead, alive, stab_soft, qual_contact.
        (terminal_su / terminal_bal은 episode 끝에 get_terminal_bonus가 별도로 추가)
    """
    # ── 1) State-based terms (bin 상태에만 의존) ─────────────
    # SU는 옵션 D로 per-step에서 빠지고 episode 종료 시 일괄 지급(get_terminal_bonus).
    # SU_WEIGHT=0이라 사실상 0이지만, 디버깅 시 SU_WEIGHT를 다시 켜는 옵션을 위해 식은 남김.
    r_su = SU_WEIGHT * float(bin_obj.SU) * 10.0
    r_dead = -1.0 * DEAD_WEIGHT * _dead_ratio(bin_obj) * 10.0

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
            # 정규화: 접촉 면적(mm²) / 아이템 측면적 합 = 4면 접촉 비율 ∈ [0, 1.0]
            # bin/item 크기 무관하게 step당 r_contact ∈ [0, CONTACT_WEIGHT].
            contact_val = _get_contact_score(bin_obj, placed_item)
            max_lateral = _item_lateral_area(placed_item)
            r_contact = CONTACT_WEIGHT * (contact_val / max(1e-6, max_lateral))
        except Exception:
            r_contact = 0.0

    # ── 3) 합산 ──────────────────────────────────────────────
    # r_bal은 per-step에서 빠짐 (terminal로 이전, get_terminal_bonus 참조).
    total_score = r_su + r_dead + r_alive + r_stability + r_contact

    terms: Dict[str, float] = {
        "eff_su":       float(r_su),
        "eff_dead":     float(r_dead),
        "alive":        float(r_alive),
        "stab_soft":    float(r_stability),
        "qual_contact": float(r_contact),
    }
    return float(total_score), terms


def get_terminal_bonus(bin_obj) -> Tuple[float, Dict[str, float]]:
    """
    에피소드 정상 종료(finished=True) 시 두 가지 final 신호를 일괄 지급.

    1) terminal_su  = TERMINAL_SU_WEIGHT  × final_SU   × 10  (∈ [0, 50])
       — "최종 SU 최대화"라는 진짜 목표와 1:1 align (옵션 D).

    2) terminal_bal = TERMINAL_BAL_WEIGHT × bal_score  × 10  (∈ [0, 5])
       — 적재 완료 시점의 무게중심 균형 점수. transport 안정성 신호.
       bal_score = 0.5 × ((1-imb_norm) + (1-I_norm)) ∈ [0, 1] (1=완벽 균형).

    env._finalize_step의 finished 분기에서 추가로 호출하여 reward에 가산.
    truncated(max_steps_reached) 케이스도 finished에 포함 → 부분 SU/bal에도 비례 보상.
    실패 종료(NOOP/RETRY_LIMIT/물리 실패)에는 호출되지 않음 → 미달 시 큰 기회비용.
    """
    if bin_obj is None:
        return 0.0, {"terminal_su": 0.0, "terminal_bal": 0.0}

    # SU 보너스
    su = float(bin_obj.SU)
    su_bonus = TERMINAL_SU_WEIGHT * su * 10.0

    # Balance 보너스 — final state의 무게중심 균형
    try:
        imb_raw, I_raw, sum_m_t = balance_feature(bin_obj, cand_item=None)
        imb_n, I_n = normalize_balance_features(imb_raw, I_raw, sum_m_t)
        # imb_n / I_n ∈ [0, 1], 0=완벽 균형 → score = 1
        bal_score = 0.5 * ((1.0 - imb_n) + (1.0 - I_n))   # ∈ [0, 1]
    except Exception:
        bal_score = 0.0
    bal_bonus = TERMINAL_BAL_WEIGHT * bal_score * 10.0

    total = su_bonus + bal_bonus
    return float(total), {
        "terminal_su":  float(su_bonus),
        "terminal_bal": float(bal_bonus),
    }


def get_failure_penalty(failure_code=None) -> Tuple[float, Dict[str, float]]:
    """
    실패 종료 시 페널티 반환. env가 직접 호출.

    실패 분류:
      ① 정책적 실패 (agent 책임) — NOOP / RETRY_LIMIT
         → STRING_PENALTIES에서 직접 값(-0.5)을 읽음. 감쇠 없음.
      ② 물리적 실패 (action masking이 잡았어야 함) — PENALTY_MAP 정수 키
         → masking 신뢰 정책으로 PENALTY_SCALE_PHYSICAL(× 0.01)로 가볍게.
      ③ 알 수 없는 케이스 — 보수적으로 default × scale.
    """
    if isinstance(failure_code, str) and failure_code in STRING_PENALTIES:
        # ① 정책적 실패: 명시적 페널티
        penalty = STRING_PENALTIES[failure_code]
    elif failure_code is not None and failure_code in PENALTY_MAP:
        # ② 물리적 실패: 가벼운 감쇠 (masking 신뢰)
        penalty = PENALTY_MAP[failure_code] * PENALTY_SCALE_PHYSICAL
    else:
        # ③ 미분류: 기본값 + 감쇠
        penalty = PENALTY_MAP.get("default", -5.0) * PENALTY_SCALE_PHYSICAL

    terms: Dict[str, float] = {k: 0.0 for k in _TERM_KEYS}
    terms["fail"] = float(penalty)
    return float(penalty), terms
