# Reward 설계 분석 및 개선 제안

> 대상 파일: `planning/RL/PalletFit_RL/reward_builder.py`
> 목적: 현재 reward 구조를 정리하고, 각 항목이 **설계 의도대로 작동하고 있는지**를 점검 → env 개선과 함께 정상화.

---

## 1. 현재 구현 구조

### 1-1. 전체 수식

**성공 step:**
```
total_reward = ALIVE_BONUS
             + r_su
             + r_dead
             + r_stability
             + r_contact
             + r_bal
             # + r_flat   ← 주석 처리됨
```

**실패 종료 step:**
```
total_reward = PENALTY_MAP[failure_code] × 0.001
```

### 1-2. 가중치 (현재 값)

| 상수 | 값 | 의미 |
|---|---|---|
| `ALIVE_BONUS` | **0.0** | 적재 성공 기본 보상 |
| `SU_WEIGHT` | 0.0 | per-step 비활성 (옵션 D, terminal로 이전) |
| `TERMINAL_SU_WEIGHT` | 5.0 | episode 종료 시 final_SU 보너스 |
| `DEAD_WEIGHT` | 0.3 | dead volume 약한 가이드 (margin-bridge는 stab/contact가 자연 상쇄) |
| `CONTACT_WEIGHT` | 1.0 | (정규화됨, was 0.1) step당 r_contact ∈ [0, 1.0] |
| `TERMINAL_BAL_WEIGHT` | 0.5 | episode 종료 시 final balance score 보너스 (was BALANCE_WEIGHT 0.5, per-step) |
| `STABILITY_PENALTY_WEIGHT` | 0.5 | 지지율 부족 soft penalty |
| `FLATNESS_WEIGHT` | 0.002 | 윗면 균일도 (미사용) |
| `CONTACT_WEIGHT` | 0.1 | 접촉 면적 보너스 |
| `BALANCE_WEIGHT` | 0.5 | 무게중심 균형 |

### 1-3. PENALTY_MAP (2026-05-02 갱신)

**정책적 실패 (`STRING_PENALTIES`, agent 책임 → 명시적 큰 페널티)**:
| failure_code | 페널티 |
|---|---|
| `"NOOP"` | **-0.5** |
| `"RETRY_LIMIT"` | **-0.5** |

**물리적 실패 (`PENALTY_MAP × PENALTY_SCALE_PHYSICAL=0.01`, masking 신뢰 → 가벼운 신호)**:
| failure_code | 설정값 | 실제값 (×0.01) |
|---|---|---|
| `FAIL_COLLISION` | -5.0 | -0.050 |
| `FAIL_OUT_OF_BOUNDS_*` | -5.0 | -0.050 |
| `FAIL_NO_TOP_EMPTY` | -5.0 | -0.050 |
| `FAIL_NO_SUPPORT_BOTTOM` | -3.0 | -0.030 |
| `FAIL_CG_OUTSIDE_SUPPORT` | -3.0 | -0.030 |
| `FAIL_CUMULATIVE_UNSTABLE` | -3.0 | -0.030 |
| `FAIL_WEIGHT_EXCEEDED` | -3.0 | -0.030 |
| `FAIL_SUPPORT_OVERLOAD` | -3.0 | -0.030 |
| `FAIL_OVERHANG_TOO_MUCH` | -2.0 | -0.020 |
| `FAIL_SUPPORT_AREA_INSUFFICIENT` | -2.0 | -0.020 |
| "default" | -5.0 | -0.050 |

> 실제로 env가 `failure_code`로 전달하는 값은 `"NOOP"`/`"RETRY_LIMIT"` 두 가지뿐. 정수 FAIL_* 코드들은 정의는 있지만 env가 retry로 흡수해 외부에 노출 안 됨.

---

## 2. 항목별 상세 및 타당성

### A. `r_su` — 공간 효율 gain (옛 핵심) ⚠️ **2026-05-02 옵션 D로 per-step 비활성화**

```python
# 이전
r_su = SU_WEIGHT * bin_obj.SU * 10.0   # 1.6 × ΔSU × 10 per step

# 현재 (SU_WEIGHT=0 → per-step 0)
r_su = 0.0
# SU 신호는 episode 끝에 일괄 지급:  TERMINAL_SU_WEIGHT × final_SU × 10  (Sec 7-3 참조)
```

**왜 빠졌나** — 큰-vs-작은 박스 ΔV 편향:
- ΔSU = item.volume / bin.volume.
- 100mm³ 박스 vs 800mm³ 박스 → ΔV가 ~512×, Δr_su가 ~512× 차이.
- 정책이 "큰 박스 먼저" 그리디로 편향. 정작 최적은 큰 박스를 전략 자리에 아껴두고 작은 박스로 빈틈을 메우는 경우가 많음.
- Late-game(작은 박스만 남음) 신호 약화 → critic 학습 불안정.

**현재 처리** — 옵션 D로 **재배치**(제거가 아님):
- `SU_WEIGHT=0`으로 per-step 신호 차단.
- 에피소드 정상 종료(finished) 시 `get_terminal_bonus(bin_obj)`가 `TERMINAL_SU_WEIGHT × final_SU × 10` 한 방으로 지급.
- 결과: 정책 학습이 **진짜 목표(최종 SU)와 1:1 align**, ΔV 편향 소멸.

---

### B. `r_dead` — 데드 볼륨 약한 가이드 ✅ **2026-05-02 컨셉 재정렬 + DEAD_WEIGHT 1.0 → 0.3**

```python
# 현재 (state-based delta)
r_dead = -DEAD_WEIGHT * _dead_ratio(bin_obj) * 10.0   # DEAD_WEIGHT = 0.3
# → 매 step delta = -0.3 × Δ(dead_ratio) × 10
```

**설계 의도 재정의** (2026-05-02 사용자와 컨셉 align):

1. **"이 step에서 새로 생긴 dead만 벌점"은 올바른 설계** (이전 docs의 ⚠️ 표시 정정):
   - state-based delta 구조에서 자동 보장 — `reward = curr - prev`이므로 step n에서 생긴 dead는 그 step에만 처벌, 다음 step부터는 baseline의 일부가 됨.
   - "각 action에 대한 신호는 그 action이 일으킨 결과만"이라는 RL credit assignment 기본 원칙과 일치.
   - 이전 임계값(`> 1e-4`) 설계도 같은 의도였고 이건 옳았음.

2. **Margin-bridge dead zone은 처벌 면제 의도** (사용자 원본 컨셉):
   - margin 위에 다른 아이템을 덮어 dead가 생기는 경우 → 정의상 필연적 + transport 안정성 ↑ trade-off.
   - 이전 임계값 `> 1e-4`는 이 의도를 거의 못 잡았음 (numerical noise filter 수준; margin-bridge로 생긴 큰 dead와 bad placement dead 모두 통과).

3. **Geometry-aware detection 대신 자연 상쇄 채택** (사용자 결정):
   - 이미 활성화된 `r_stability` + `r_contact`가 placement의 "good vs bad" 신호를 충분히 줌.
   - **margin-bridge** placement는 (a) 두 지지 아이템 위에 걸쳐 support_ratio≈1.0 → r_stability=0, (b) 두 아이템과 접촉 → r_contact 큼.
   - **bad floating** placement는 (a) 한 모서리 살짝 걸침 → support_ratio=0.6 → r_stability=-0.2, (b) contact 작음.
   - 결과적으로 dead 페널티가 약해도 stab/contact 차이가 둘을 잘 구분함.

4. **DEAD_WEIGHT 1.0 → 0.3**:
   - r_dead를 "약한 가이드"로 격하 (절반 이하).
   - 진짜 placement 평가는 stab/contact가 담당.
   - 별도 geometry 검출 코드 없이 컨셉 align 달성.

**비교 (1m³ bin, margin-bridge vs bad floating placement, 추정)**:

| placement 유형 | r_stability | r_contact | r_dead (delta) | 합 |
|---|---|---|---|---|
| Margin-bridge (지지 2개) | ~0 (완벽) | ~+5 (큰 접촉) | ~-0.05 (약간) | **~+4.95** |
| Tight (dead 0) | ~0 | ~+5 | 0 | **+5.00** |
| Bad floating (지지 약함) | -0.2 | ~+1 | ~-0.2 | **+0.6** |

→ margin-bridge가 페널티를 받지만 stab/contact 보너스가 압도적으로 커서 net + 보상. 컨셉과 일치.

**아직 미해결 (장기)**:
- 진짜로 margin-bridge가 학습에서 회피되는 게 관찰되면 옵션 ①(geometry-aware bridging detection)으로 advance 검토. 그땐 `bin.render()`로 placement 패턴 시각 디버깅 가능.

---

### C. `r_stability` — 지지율 soft penalty ✅ **활성화됨 (2026-04-27)** + ⚡ 연산 가속 (2026-05-02, Sec 7-6)

```python
if placed_item is not None:
    support_ratio = _calculate_support_ratio_geometric(bin_obj, placed_item)
    r_stability = (support_ratio - 1.0) * STABILITY_PENALTY_WEIGHT  # 0.5
```

**설계 의도**:
- Action Masking이 **최소 지지율**(예: 0.6)을 이미 보장한 상태.
- 여기서는 **"얼마나 완벽히(1.0)에 가까운가"**를 추가로 보상.
- 지지율=1.0 → 0점 (완벽), 지지율=0.6 → -0.2점 (감점).

**Margin-bridge 영향 (사용자 컨셉 align 결정)**:
- 함수가 측정하는 건 "기하학적 지지율" — 두 지지 아이템 사이 margin gap이 있으면 그 영역만큼 미지지로 잡힘.
- 예: 1m³ bin, B=500×500, margin=50mm → support_ratio=0.90 → r_stability=-0.05 (gap 비율에 비례).
- 의도된 동작: **물리적으로만 판단, 실제로 gap 위 영역은 미지지가 맞음**. 사용자 결정: "선택은 agent가 하는 게 맞는 것 같아. 물리적으로만 판단하자."
- 자연 상쇄: bridging은 두 아이템과 접촉 → r_contact가 큰 +로 들어와 net은 보통 +.

**현재 상태**:
- ✅ `_finalize_step`이 `placed_item`을 그대로 `build_reward`로 전달 → 정상 동작.
- ✅ `_ep_reward_terms["stab_soft"]`로 분해 로깅됨(Sec 7-2 delta 누적 fix 이후 의미 정확).
- ⚡ `_calculate_support_ratio_geometric` 호출 비용 ~0.9μs/call (이전 ~50-100μs, ~50-100× 단축). `Item.update_face_cache`/`getVertices` 캐싱 cascade 효과(Sec 7-6).

---

### D. `r_contact` — 접촉 면적 보너스 ✅ **활성화 (2026-04-27)** + ✅ **스케일 정규화 (2026-05-02, Sec 7-7)**

```python
if placed_item is not None:
    contact_val = _get_contact_score(bin_obj, placed_item)        # 4 측면 접촉 mm²
    max_lateral = _item_lateral_area(placed_item)                  # 2 × D × (W + H)
    r_contact = CONTACT_WEIGHT * (contact_val / max(EPS, max_lateral))   # ∈ [0, 1.0]
```

**설계 의도**: 주변 아이템/벽과 접촉 면적이 클수록 보너스 → 밀착 배치 유도.

**현재 상태**:
- ✅ `placed_item` 전달 경로 복원 후 매 placement에서 정상 가중.
- ✅ **스케일 정규화 완료 (2026-05-02)**: 측면적 합 기준 비율 → step당 `r_contact ∈ [0, 1.0]` 상한, bin/박스 크기 무관.
- 비중 정렬: terminal_su(64%) > alive(20%) > contact(15%) — 옵션 D 신호 hierarchy 회복.

---

### E. `r_bal` → `terminal_bal` — 무게중심 균형 ✅ **2026-05-02 terminal로 이전 (Sec 7-8)**

**현재 구현 (terminal 일괄 지급)**:
```python
# get_terminal_bonus(bin_obj) 안에서, episode 정상 종료(finished)에만 호출
imb_raw, I_raw, sum_m_t = balance_feature(bin_obj, cand_item=None)
imb_n, I_n = normalize_balance_features(imb_raw, I_raw, sum_m_t)
bal_score = 0.5 * ((1.0 - imb_n) + (1.0 - I_n))   # ∈ [0, 1] (1 = 완벽 균형)
terminal_bal = TERMINAL_BAL_WEIGHT * bal_score * 10  # ∈ [0, 5]
```

**이전 design**: `r_bal = BALANCE_WEIGHT × balance_term_capped(r_su or 1.0, bin, cand_item=None)` per-step. ±0.025/step, 누적 -0.016 (사실상 0%, 학습 신호 무영향).

**근거**: "transport 균형"은 매 step 변화가 아니라 **적재 완료 시점의 결과**. terminal 일괄 지급이 의미상 정확하고, 옵션 D(SU per-step → terminal) 철학과 일관.

**이전 두 우려 해소 상황**:

| 이전 우려 | 해소 방식 |
|---|---|
| ⚠️ **`cand_item=None`이라 incremental 효과 미측정** — 방금 놓은 아이템이 균형에 어떻게 기여했는지 모름. | ✅ **자동 해소**. terminal에선 episode 끝의 final state만 평가하면 충분 → "어떤 step에서 어디 놓았는지의 incremental 효과"를 굳이 분리할 필요가 사라짐. PPO의 credit assignment(GAE)가 final terminal_bal을 시작 step까지 backprop해주므로 incremental 효과는 학습 알고리즘 쪽에서 자동 처리. |
| ⚠️ **`cap = 0.1 × \|r_su\|`라 학습 초반에 cap≈0 → 거의 안 먹힘**. 옵션 D(SU_WEIGHT=0) 후엔 항상 cap≈0 위험. | ✅ **완전 제거**. `balance_term_capped`의 cap 식 의존을 끊고 `terminal_bal = 0.5 × bal_score × 10`로 SU와 무관한 고정 스케일. 이전 호출자가 우회용으로 `1.0` 고정 전달하던 sloppy fix도 사라짐. magnitude ∈ [0, 5]가 episode마다 일관되게 적용되어 학습 초반·후반 균등하게 작동. |

**balance score 분리 검증** (가짜 시나리오):
- 대각선 두 모서리 (대칭, 응집X): imb_n=0, I_n=0.64 → score=0.68 → terminal_bal=+3.40
- 중앙 응집:                    imb_n=0, I_n=0    → score=1.00 → terminal_bal=+5.00
- 직관(중앙 응집이 더 좋은 균형)과 일치 ✓

**현재 비중** (TSG 32 items 정상 완주): terminal_bal +4.22 = ep_return의 5.1% — 이전 0% → **5%로 의미 있는 학습 신호로 격상**.

---

### F. ~~`r_flat`~~ — ❌ **2026-05-02 영구 제거 (Sec 7-9)**

이전: 주석 처리 상태로 잔존(코드는 존재, 호출 안 됨).
현재: `FLATNESS_WEIGHT` 상수 + `_get_surface_std` 헬퍼 + `collect_ez_stats` import 모두 삭제.

**제거 근거**:
1. **obs와 정보 중복**: `globals.h_score`가 이미 height distribution을 policy에 노출. 보상 신호로 또 줄 필요 없음.
2. **다른 신호와 중복**: SU/dead/contact가 평탄도와 자연 상관. 평평하게 쌓으면 SU↑ → terminal_su가 implicit 보상.
3. **옵션 D 철학 충돌**: per-step proxy 신호 추가는 "최종 목표만 직접 평가" 원칙에 역행.
4. **forward-looking은 critic의 일**: "이 상태가 미래에 좋다"는 PPO value function이 학습할 영역. reward에 박으면 critic 의미 왜곡.
5. **misdirection 위험**: 작은 박스를 굳이 위로 올려 평탄도 맞추는 행동이 진짜 SU와 충돌 가능.
6. **이미 비활성 + 무영향**: 이전 실험에서 제거해도 학습 차이 없었다는 사용자 판단.

**재도입 트리거** (학습 관찰 후 다시 본다면):
- eval에서 SU=1.0인데 윗면이 들쭉날쭉해서 다음 batch 적재 어려운 시나리오 발견.
- 정책이 "큰 박스 한쪽 쌓기"만 고집하고 평탄도 무시 패턴.
- → 그땐 per-step `r_flat`이 아니라 `terminal_flat = TERMINAL_FLAT_WEIGHT × (1 - std/max_std) × 10` (terminal 일괄 지급, 옵션 D 일관) 형태로 도입.

---

### G. `ALIVE_BONUS` = 0.5 ✅ **2026-05-02 옵션 D 일환으로 활성화**

**현재 의미** — Per-placement 정액 dense 보상:
- 옵션 D로 per-step `r_su`가 빠지면서 dense gradient용 신호가 필요해짐.
- 모든 placement에 동일한 `+0.5` → 큰-작은 박스 차별 없이 "뭐든 놓으면 +" 신호.
- 작은 박스 무시 방지(원 ALIVE_BONUS 도입 의도와 동일하지만 이번엔 비활성 → 활성).

**튜닝 가이드**:
- 너무 크면 SU 무관 placement 남발 가능(품질 저하 감수하고 개수만 늘림).
- 너무 작으면 NOOP/early-stop 빈도 증가.
- 32 step 에피소드 기준 alive 누적 = 16.0, terminal_su(SU=1.0) = 50.0 → terminal이 ~3× 더 큰 비중.
- 현재 0.5는 alive:terminal ≈ 1:3 비율로 "끝까지 가야 한다"는 신호가 우세하도록 설정됨.

---

### H. 실패 페널티 — ✅ **2026-05-02 옵션 D 후속으로 분리/강화** (이전: ⚠️ 사실상 0)

**이전 문제**: 모든 실패에 `× 0.001` 일괄 감쇠 → NOOP -5e-6, 물리 실패 -0.005. 옵션 D로 per-step SU 신호가 빠진 직후엔 정책이 NOOP을 명시적으로 회피할 신호가 너무 약했음.

**설계 의도 (현재)**: 실패 종류를 두 카테고리로 나눠 차등.

```python
# 정책적 실패 (agent 책임) — 명시적 큰 페널티
STRING_PENALTIES = {
    "NOOP":        -0.5,
    "RETRY_LIMIT": -0.5,
}

# 물리적 실패 (action masking이 잡았어야 함) — 가벼운 감쇠 유지
PENALTY_SCALE_PHYSICAL = 0.01    # was implicit × 0.001 → × 0.01 (10× 강화)

def get_failure_penalty(failure_code=None):
    if isinstance(failure_code, str) and failure_code in STRING_PENALTIES:
        penalty = STRING_PENALTIES[failure_code]
    elif failure_code in PENALTY_MAP:
        penalty = PENALTY_MAP[failure_code] * PENALTY_SCALE_PHYSICAL
    else:
        penalty = PENALTY_MAP.get("default", -5.0) * PENALTY_SCALE_PHYSICAL
    ...
```

**왜 이 구분이 맞나**:
- `"NOOP"`/`"RETRY_LIMIT"`는 **policy 결정의 직접 결과** — agent가 NOOP을 골랐거나, retry 한계까지 nyukin 갈 만큼 부적절한 placement를 반복 선택했음. 명시적 페널티가 학습 신호.
- 물리적 실패(`FAIL_*`)는 **action masking이 1차 방어**해야 함. mask가 새는 edge case에 한정해 가벼운 신호만 줘서 masking 신뢰 정책 유지.

**값 산정 근거** (옵션 D 기준):
- per-placement alive = +0.5, 정상 완주 ep_return ≈ +125 (TSG 32-item).
- NOOP/RETRY = -0.5는 **1 placement 분의 명시적 손실** + terminal_su 미지급(~50)으로 합쳐 거대한 회피 신호.
- 즉시 NOOP의 ep_return = -0.5 vs 정상 완주 +125.98 → **차이 126.5점**.
- 너무 크게 잡으면 초기 exploration이 NOOP 한 번이라도 골랐을 때 학습이 출렁임 → -0.5가 균형점.

**검증**:
| failure_code | 이전 | 현재 | 배율 |
|---|---|---|---|
| `"NOOP"` | -5e-6 | **-0.5** | **100,000×** |
| `"RETRY_LIMIT"` | -5e-6 | **-0.5** | 100,000× |
| 물리 fail (FAIL_*) | -0.005 ~ -0.002 | -0.05 ~ -0.02 | 10× |
| Unknown / default | -0.005 | -0.05 | 10× |

---

## 3. 항목 활성화 현황 (2026-05-02 갱신)

| 항목 | 설계값 | 실제 동작 | 비고 |
|---|---|---|---|
| `r_su` (per-step) | — | ⚠️ **비활성화** (SU_WEIGHT=0) | 옵션 D로 terminal로 이전 |
| `terminal_su` (NEW) | 5 × final_SU × 10 | ✅ 정상 | finished 종료에만 일괄 지급 |
| `ALIVE_BONUS` (per-placement) | 0.5 (정액) | ✅ **활성화** | 옵션 D dense gradient |
| `r_dead` | -0.3 × dead_ratio × 10 | ✅ 정상 (state delta, 약한 가이드) | margin-bridge는 stab/contact가 자연 상쇄 |
| `r_stability` | soft penalty [-0.2, 0] | ✅ 활성화 | `placed_item` 전달 |
| `r_contact` | 밀착 보너스 | ✅ 활성화 | 스케일 정규화는 미적용 |
| `terminal_bal` (NEW) | 0.5 × final balance × 10 | ✅ episode 종료에만 일괄 지급 | per-step `r_bal`에서 이전 (Sec 7-8) |
| ~~`r_flat`~~ | — | ❌ 영구 삭제 | obs.h_score와 정보 중복, 옵션 D 철학 충돌 (Sec 7-9) |
| 정책 실패 페널티 (NOOP/RETRY) | — | ✅ **-0.5 명시 적용** | STRING_PENALTIES |
| 물리 실패 페널티 (FAIL_*) | -5 ~ -2 | ✅ -0.05 ~ -0.02 | × 0.01로 완화 (was × 0.001) |

**현재 학습 신호 구조** (2026-05-02 기준):
```
ep_return ≈ ALIVE_BONUS × n_placements          (per-step dense, 작은 신호)
          + terminal_su = 50 × final_SU         (finished 종료에만, 가장 큰 신호)
          + r_stability + r_contact             (placement-specific 보너스)
          + r_dead + r_bal                      (state delta, 약함)
          + (실패 시) fail × 0.001               (사실상 0)
```
TSG 32-item 기준 측정값: alive 16 + terminal_su 50 + qual_contact 60 + bal -0.016 = **+125.98** (정상 완주).

---

## 4. 개선 제안

### 4-1. 즉시 수정 (env.py와 함께)

#### (a) `cand_item` 전달 경로 복원

env.py 수정 (Task 2 B번)으로 `placed_item`이 reward까지 도달하면:
- `r_stability` 자동 활성화 → action masking의 최소 지지율(0.6~) 위에서 1.0 완벽 지지를 지향하는 soft penalty 정상화.
- `r_contact` 자동 활성화 → 다만 스케일(/10000) 재검토 필요.

#### (b) 실패 페널티 감쇠 완화

```python
# 현재
penalty *= 0.001

# 제안: 실패 종류별 차등
PENALTY_SCALE = {
    "NOOP":        0.1,    # NOOP 선택: 비교적 강한 페널티
    "RETRY_LIMIT": 0.05,   # retry 한계 초과: 중간
    "default":     0.02,   # 물리 실패: 약한 페널티 (masking 신뢰)
}
```

구체 수치는 `r_su` 평균 크기와 비교하여 조정.
일단 **`× 0.001` → `× 0.05`** 로 한 자릿수 올려보는 게 안전한 출발점.

#### (c) `r_contact` 스케일 정규화

```python
# 현재 (하드코딩)
r_contact = CONTACT_WEIGHT * (contact_val / 10000.0)

# 제안 (아이템 접촉면 대비 정규화)
if cand_item is not None:
    w, h, d = cand_item.getDimension()
    # 가능한 최대 접촉 면적 (bottom + 4 sides)
    max_contact = 2 * (w*d + w*h + h*d)
    r_contact = CONTACT_WEIGHT * (contact_val / max(eps(), max_contact))
```

### 4-2. 구조 개선

#### (d) `r_bal`에 `cand_item` 전달

```python
# 현재
bal_term_val = balance_term_capped(r_su, after_bin, cand_item=None)

# 개선
bal_term_val = balance_term_capped(r_su, after_bin, cand_item=placed_item)
```

"이 아이템을 놓았을 때의 balance score"를 계산하게 됨.

#### (e) 완성(complete) 보너스

현재는 "모든 아이템을 다 쌓음"과 "적당히 쌓고 NOOP"의 차이가 `no_op_selected` 페널티뿐.

```python
# 제안
if terminated and finished and terminal_reason == "no_items":
    complete_bonus = COMPLETE_BONUS_WEIGHT * after_bin.SU   # ex: 2.0 × SU
    total_reward += complete_bonus
```

### 4-3. 재고할 항목

#### (f) `r_flat` 부활 여부

SU/dead/balance와의 중복을 실험으로 확인한 후 결정:
- Option 1: 제거 유지 (주석/삭제) → 깔끔
- Option 2: 가중치 상향(0.002 → 0.02) + globals에서 h_score를 제거 (중복 신호 정리)
- Option 3: 작은 아이템 배치 시에만 가중치 부여 (아이템 부피 역비례)

#### (g) `ALIVE_BONUS`

작은 아이템 방치 현상이 보이면 0.01~0.05로 소량 부활. 현재 로그에서 작은 아이템 적재율을 우선 측정.

---

## 5. 권장 적용 순서 (상태 갱신: 2026-05-02)

| 순서 | 항목 | 난이도 | 기대 효과 | 상태 |
|---|---|---|---|---|
| 1 | **build_reward 시그니처: state-based로 전환** | 쉬움 | env deepcopy 제거 | ✅ 적용 (Sec 7-1) |
| 2 | **`placed_item` 전달 (env와 묶음)** | 쉬움 | stability/contact 활성화 | ✅ 적용 (Sec 7-1) |
| 8 | **`_ep_reward_terms` per-step delta 누적으로 변경** | 쉬움 | TB term 분해가 진짜 reward 기여도와 일치 | ✅ 적용 (Sec 7-2) |
| 9 | **옵션 D: SU per-step 제거 + ALIVE_BONUS=0.5 + terminal SU 보너스** | 중간 | 큰-작은 박스 ΔV 편향 제거, 진짜 목표(final SU)와 align | ✅ 적용 (Sec 7-3) |
| 3 | **실패 페널티 분리/강화 (NOOP/RETRY = -0.5, 물리 × 0.01)** | 매우 쉬움 | NOOP 남발 억제, 정책/물리 실패 차등 | ✅ 적용 (Sec 7-4) |
| 10 | **r_dead 컨셉 재정렬 + DEAD_WEIGHT 1.0 → 0.3** | 매우 쉬움 | margin-bridge 자연 상쇄, dead는 약한 가이드로 격하 | ✅ 적용 (Sec 7-5) |
| 11 | **Item.update_face_cache / getVertices 캐싱** | 쉬움 | r_stability 함수 ~50-100× 가속, get_direction_overlap 등 cascade | ✅ 적용 (Sec 7-6) |
| 4 | **`r_bal` terminal로 이전** | 쉬움 | "final balance" 의미 정확, 옵션 D 철학 일관 | ✅ 적용 (Sec 7-8) |
| 5 | **`r_contact` 스케일 정규화 + CONTACT_WEIGHT 0.1 → 1.0** | 쉬움 | bin/박스 크기 무관, 신호 hierarchy 회복 | ✅ 적용 (Sec 7-7) |
| 6 | ~~완성 보너스 추가~~ | — | — | ✅ 옵션 D에 흡수 |
| 7 | ~~`r_flat` 재검토~~ → 영구 삭제 결정 | — | 정보 중복, 옵션 D 철학 충돌 | ✅ 삭제 (Sec 7-9) |

### 5-1. 다음 권장 우선순위 (2026-05-02 시점)

> 코드 미적용 — 학습 결과 관찰 후 결정용 메모.

**(a) ⑤ `r_contact` 스케일 정규화 — 우선도 높음**
- 위치: `reward_builder.py:216`, `r_contact = CONTACT_WEIGHT * (contact_val / 10000.0)`.
- 현상 우려: 디버그 트레이스에서 step당 `qual_contact` 기여가 +1.25 ~ +5.0으로 들어옴(eff_su=0이 된 옵션 D 이후 placement-specific 항목 중 가장 큰 비중). 32-item 에피소드에서 `qual_contact` 누적 ≈ +60이 terminal_su(+50)보다 큼.
- 시급성 트리거(이 신호가 보이면 즉시 적용): TB의 `train/reward_ep/qual_contact` 가 `train/reward_ep/terminal_su`보다 빨리 천장 치고 SU는 안 오를 때 → 정책이 contact만 챙기고 SU를 무시하는 학습 함정.
- 제안: `(contact_val / max_contact)` — `max_contact = 2 * (w*d + w*h + h*d)` 등 bin/item 크기 비례 정규화. 기댓값을 [0, 1]로 묶어 `qual_contact ∈ [0, CONTACT_WEIGHT]` 안정 범위로.

**(b) ④ `r_bal`에 `cand_item` 전달 — 우선도 중간**
- 위치: `reward_builder.py:195`, `balance_term_capped(1.0, bin_obj, cand_item=None)`.
- 현재: 전역 balance만 측정 → 같은 step에서 어디에 놓았는지가 신호에 반영 안 됨. `r_bal` 항이 매우 작아(±0.02 수준) 학습에 직접 영향은 적음.
- 시급성 트리거: 무게중심이 한쪽으로 쏠리는 패턴이 평가에서 관찰될 때.
- 제안: `cand_item=placed_item`으로 변경. 단 `balance_feature(bin, cand_item=X)`가 X를 포함한 가정으로 잘 동작하는지 호출부 확인 필요.

**(c) ⑦ `r_flat` 재검토 — 우선도 낮음 (실험 후)**
- 현재: 주석 처리. obs의 `globals.h_score`가 같은 정보를 들고 있으나 reward 신호는 없음.
- 학습 결과에서 "계단형 스택"(평탄도 떨어지는 형태)이 자주 보이면 가중치 0.01~0.02로 부활 검토.

**(d) 파라미터 튜닝 — 학습 결과 관찰 후**
- `TERMINAL_SU_WEIGHT=5.0` ↔ `ALIVE_BONUS=0.5`: 현재 비율 alive:terminal ≈ 1:3. terminal이 약하면 (final SU에 둔감하면) TERMINAL_SU_WEIGHT를 7~10으로 상향. 너무 강하면 (NOOP보다 random placement만 노리면) ALIVE_BONUS를 0.3으로 하향.
- `STRING_PENALTIES` (현재 -0.5): 초기 NOOP 빈도가 높으면 -1.0으로 강화. 학습이 출렁이면 -0.3으로 완화.

**관찰 포인트** (학습 시작 후 TB로 확인):
- `train/reward_ep/terminal_su` 평균이 정상 완주 기준 ~50 근처에서 saturate 하는지.
- `train/reward_ep/qual_contact` 가 terminal보다 먼저 천장 치는지 (→ ⑤ 시급).
- `train/reward_ep/fail` 평균 절댓값이 -0.1 미만으로 빠르게 줄어드는지 (→ NOOP 회피 학습 진행 중).
- `eval/SU` 곡선이 단조 증가인지 — 평탄/하락 시 weight 균형 문제.

---

## 6. 요약

```
현재 reward (2026-05-02, 옵션 D 적용 후):
  per-step:
    + ALIVE_BONUS (0.5)               × 모든 placement
    + r_dead (state delta, 약한 가이드 — DEAD_WEIGHT=0.3)
    + r_stability (placement-specific)
    + r_contact (placement-specific, 정규화 ∈ [0, 1.0])
  episode 종료(finished):
    + terminal_su  = 5.0 × final_SU      × 10  ← SU 신호의 본진
    + terminal_bal = 0.5 × bal_score     × 10  ← transport 균형
  실패 종료:
    + NOOP / RETRY_LIMIT  → -0.5 (명시적, 100,000× 강화)
    + 물리 fail (FAIL_*)   → -0.05 ~ -0.02 (10× 강화, masking 신뢰는 유지)

→ "최종 SU 최대화"가 정책 학습 신호와 1:1 align.
→ 큰-작은 박스 ΔV 편향 제거.
→ 실패 시 terminal_su 미지급(~50점) + 명시 페널티(-0.5) 합쳐 총 ~125점 회피 신호.

남은 fix: 없음 — roadmap의 모든 항목 적용 완료 (r_flat은 영구 삭제 결정).
```

---

## 7. 적용된 변경사항

### 7-1. State-based `build_reward` + `placed_item` 전달 (2026-04-27)

`reward_builder.py`:
- 시그니처 `build_reward(before_bin, after_bin, cand_item)` → **`build_reward(bin_obj, placed_item=None)`**.
- 반환: `(total_score, terms)` — bin 상태의 절대 점수 + 항목 분해.
- env에서 step n과 n+1의 점수 차이(`reward = curr - prev`)로 사용 → **deepcopy 불필요**.
- placement-specific 항목(`alive`, `stab_soft`, `qual_contact`)은 `placed_item`이 있을 때만 가산.
- 실패 종료는 `get_failure_penalty(failure_code)`로 분리 (terms엔 `fail` 키만).

`env.py`:
- `_finalize_step`에서 성공 분기 시 `placed_item`을 그대로 `build_reward`에 전달 → `r_stability`, `r_contact` 활성화.
- `_prev_score = curr_score - placement_only` 로직으로 placement-specific 보너스가 다음 step의 baseline에 누적되지 않게 함(일회성 유지).

→ 항목 활성화 현황(Sec 3) 표대로 5개 항목이 학습 신호에 반영되기 시작.

### 7-2. `_ep_reward_terms` per-step delta 누적으로 변경 (2026-05-01)

**문제 진단**:
- `terms`는 절대 score 분해(예: `eff_su = SU_WEIGHT × bin.SU × 10`)인데, 기존 코드가 매 step `_ep_reward_terms[k] += terms[k]`로 누적했음.
- 결과: `sum(_ep_reward_terms.values()) ≠ ep_return`. 디버그 측정 사례 — 한 에피소드에서 `ep_return = +75.98`인데 term 합은 `+340.79` (≈ 4.5× 괴리).
- 이 dict는 `info[f"r_ep_{k}"]`를 통해 `HistoryCollectorCallback`이 TB의 `train/reward_ep/*`로 로깅 → **잘못된 reward 분해를 학습 모니터링에 노출하고 있었음**.

**수정 (env.py)**:
- `__init__`: `self._prev_terms: Dict[str, float] = {}`, `self._PLACEMENT_TERMS = ("alive", "stab_soft", "qual_contact")` 신설.
- `reset()`에서 initial `build_reward(self._bin, placed_item=None)`의 terms를 `_prev_terms`로 저장(state-only baseline).
- `_finalize_step()`:
  - **성공/finished 경로**: `delta_terms[k] = terms[k] - _prev_terms.get(k, 0.0)`. state-based 항목은 차이만 누적되고, placement-specific 항목은 `_prev_terms`에 0이 들어 있어 delta가 그 step의 전액과 같아짐 → reward 분해와 정확히 일치.
  - 그 다음 `_prev_terms = {k: 0 if k in PLACEMENT else terms[k]}` 로 갱신(`_prev_score`와 동일 원칙).
  - **실패 경로**: `get_failure_penalty`가 반환한 terms는 이미 per-step 기여도(`fail`만 nonzero)라 그대로 누적, `_prev_terms`/`_prev_score`는 갱신 안 함.

**검증 (smoke test)**:
- 32 step 성공 에피소드에서 `sum(_ep_reward_terms.values()) == _ep_return == +75.984375` (Δ = 0).
- 매 step `sum(delta_terms) == reward` 일치 (mismatch 0건).
- 실패 경로(NOOP) `terms["fail"] == _ep_return == -0.000005`.
- term 분해 의미 변화 예시:
  | term | 이전 (절대 누적) | 지금 (delta 누적) |
  |---|---|---|
  | `eff_su` | +280.0 | **+16.0** (= `(SU_final − SU_init) × 16`) |
  | `bal` | +0.79 | **−0.016** |
  | `qual_contact` | +60.0 | +60.0 (placement-specific는 그대로) |
  | 합 | 340.79 (≠ ep_return) | **75.98 (= ep_return)** |

**효과**:
- `train/reward_ep/eff_su` 같은 TB 메트릭이 **에피소드에서 받은 진짜 기여도**가 됨 → 어느 항목이 학습 신호의 몇 %를 만들었는지 직접 비교 가능.
- HistoryCollectorCallback의 키/경로는 그대로(같은 dict 참조), 의미만 정확해짐 → 다운스트림 분석 코드 변경 불필요.
- 단, 학습 비교 그래프에서 시점 전후로 `train/reward_ep/*` 절댓값이 달라짐(스케일이 줄어듦). reward 자체와 ep_return은 변동 없음.

### 7-3. 옵션 D — SU per-step 제거 + ALIVE_BONUS 활성화 + terminal SU 보너스 (2026-05-02)

**문제 진단** — 큰-vs-작은 박스 ΔV 편향:
- ΔSU = item.volume / bin.volume.
- 1m³ bin 기준: 100mm 박스 vs 800mm 박스 → ΔV가 **~512×** 차이, Δr_su도 동일 비율 차이.
- 정책이 "큰 박스 먼저" 그리디로 편향 → 최적은 "큰 박스를 전략 자리에 아껴두고 작은 박스로 빈틈 메우기"인 시나리오에서 손해.
- Late-game(작은 박스만 남음) 신호가 묻혀 critic 학습 분산 ↑.
- 에피소드별 박스 크기 분포만으로 ep_return이 흔들려 학습 곡선 해석 어려움.

**옵션 D 설계**:
- SU 신호를 **재배치**(제거 아님): per-step 분산 → episode 종료 일괄 지급.
  - per-step `r_su = SU_WEIGHT × bin.SU × 10` → `SU_WEIGHT = 0`으로 차단.
  - finished 종료 시 `terminal_su = TERMINAL_SU_WEIGHT × final_SU × 10` 한 방.
- Per-placement dense gradient는 `ALIVE_BONUS = 0.5` 정액으로 유지 → 작은 박스도 "놓으면 +0.5".
- 결과: 정책이 "최종 SU 최대화" 자체를 학습 목표로 → 평가 지표와 1:1 align.

**수정 (`reward_builder.py`)**:
```python
ALIVE_BONUS         = 0.5   # was 0.0 — per-placement 정액 보상 활성화
SU_WEIGHT           = 0.0   # was 1.6 — per-step 분배 차단 (수식은 보존)
TERMINAL_SU_WEIGHT  = 5.0   # NEW — episode 종료 시 final_SU 보너스 가중치

def get_terminal_bonus(bin_obj):
    su = float(bin_obj.SU) if bin_obj is not None else 0.0
    bonus = TERMINAL_SU_WEIGHT * su * 10.0
    return float(bonus), {"terminal_su": float(bonus)}
```
- `_TERM_KEYS`에 `"terminal_su"` 추가 (TB 로깅용).
- `get_failure_penalty`는 그대로 → 실패 경로는 terminal 미지급(큰 기회비용 발생).

**수정 (`env.py`)**:
- `from ... import get_terminal_bonus` 추가.
- `_finalize_step` 성공/finished 분기 끝에:
  ```python
  if (terminated or truncated) and finished:
      t_bonus, t_terms = get_terminal_bonus(self._bin)
      reward += t_bonus
      for k, v in t_terms.items():
          delta_terms[k] = delta_terms.get(k, 0.0) + float(v)
  ```
- delta_terms에 합쳐 누적되므로 `sum(_ep_reward_terms.values()) == _ep_return` 정합 유지(7-2).

**검증 (3 시나리오)**:

| 시나리오 | placed | SU | ep_return | terminal_su | alive 누적 | 검증 |
|---|---|---|---|---|---|---|
| 정상 완주 (TSG 32개) | 32 | 1.000 | **+125.98** | +50.00 | +16.00 | sum == return ✓, terminal = 5×1×10 ✓, alive = 0.5×32 ✓ |
| 즉시 NOOP 실패 | 0 | 0.000 | −5e-6 | 0 | 0 | terminal 미지급 ✓ (finished=False) |
| 5번 적재 후 NOOP | 5 | 0.156 | +4.96 | 0 | +2.50 | 부분 SU여도 실패 경로 → terminal 미지급 (50점 기회비용) |

**행동 인센티브 분석**:
- 정상 완주의 ep_return ≈ `0.5 × n_placements + 50 × SU + (contact ~60) + small`.
- TSG 균일 32개 / SU=1.0 기준: alive 16 + terminal 50 + contact 60 = ~125 → terminal 비중 **~40%**.
- 5번 NOOP의 ep_return ≈ 2.5 → 정상 완주 대비 **120점 기회비용**. NOOP 페널티(-5e-6)는 무의미하지만 **기회비용 자체가 거대한 페널티 역할**을 함.
- 이론상 GAE 영향: gamma=0.999, 30 step 에피소드에서 t=0 시점의 terminal_su는 0.999³⁰ ≈ 0.97로 거의 그대로 backprop. 100 step도 0.90으로 양호.

**다음 fix가 더 시급해진 이유**:
- 옵션 D 적용 후 per-step 신호 분산이 줄어 fail penalty(현재 `× 0.001`로 사실상 0)가 NOOP/early-stop을 직접 막지 못함.
- 다행히 terminal_su 기회비용이 그 역할을 대신하지만, 명시적 fail penalty 강화(roadmap ③)도 같이 가면 학습 안정성 ↑. 우선순위 격상 권장.

**효과**:
- 큰-작은 박스 ΔV 편향 소멸(모든 placement = +0.5).
- 정책 학습 신호와 평가 지표(final SU)가 일치.
- 에피소드별 ep_return 변동이 줄어들어(주로 SU에 비례) 학습 곡선이 깔끔해질 것으로 예상.
- TB의 `train/reward_ep/eff_su`는 0이 되고, 새로 `train/reward_ep/terminal_su`가 나타남(HistoryCollectorCallback이 `info["r_ep_*"]`를 자동 로깅하므로 코드 변경 불필요).

### 7-4. 실패 페널티 분리/강화 — `STRING_PENALTIES` 도입 (2026-05-02)

**배경**:
- 옵션 D(7-3)로 per-step `r_su` 신호가 빠지면서, 정책이 NOOP을 명시적으로 회피할 신호가 약해질 위험.
- 기존 `× 0.001` 일괄 감쇠는 NOOP=-5e-6, 물리 fail=-0.005로 사실상 0 → roadmap ③ 시급도 격상.
- env가 실제로 전달하는 `failure_code`는 `"NOOP"`/`"RETRY_LIMIT"` 두 가지뿐(정수 FAIL_* 코드는 env가 retry로 흡수). 따라서 두 카테고리로 분리해 차등 적용이 정확.

**수정 (`reward_builder.py`)**:
```python
STRING_PENALTIES = {
    "NOOP":        -0.5,    # 자발적 종료 — agent 책임, 명시 페널티
    "RETRY_LIMIT": -0.5,    # 물리 재시도 한계 초과 — 나쁜 placement 반복 선택
}
PENALTY_SCALE_PHYSICAL = 0.01    # was implicit × 0.001 → × 0.01 (10× 강화)

def get_failure_penalty(failure_code=None):
    if isinstance(failure_code, str) and failure_code in STRING_PENALTIES:
        penalty = STRING_PENALTIES[failure_code]              # ① 정책 실패
    elif failure_code is not None and failure_code in PENALTY_MAP:
        penalty = PENALTY_MAP[failure_code] * PENALTY_SCALE_PHYSICAL  # ② 물리 실패
    else:
        penalty = PENALTY_MAP.get("default", -5.0) * PENALTY_SCALE_PHYSICAL  # ③ 미분류
    ...
```

**값 산정 근거**:
- per-placement alive = +0.5, 정상 완주 ep_return ≈ +125 (TSG 32-item 기준).
- NOOP/RETRY = -0.5는 "1 placement 분 명시 손실" + terminal_su 미지급(~50)으로 합쳐 ~50점 회피 신호.
- 너무 크게 잡으면 초기 exploration 시 NOOP 한 번에 학습이 출렁임 → -0.5가 균형점.
- 물리 fail은 masking이 잡아야 하는 영역이라 가벼운 신호 유지(× 0.01).

**검증 (4 시나리오)**:

| 시나리오 | failure_code | penalty 이전 → 현재 | 배율 |
|---|---|---|---|
| 즉시 NOOP | `"NOOP"` | -5e-6 → **-0.5** | 100,000× |
| 재시도 한계 | `"RETRY_LIMIT"` | -5e-6 → -0.5 | 100,000× |
| FAIL_COLLISION (실제로는 도달 안 함) | `-3` (int) | -0.005 → -0.05 | 10× |
| 미분류 (`None` / unknown str) | — | -0.005 → -0.05 | 10× |

**행동 인센티브 비교 (TSG 32-item, 옵션 D + 7-4 적용 후)**:
- 즉시 NOOP: ep_return = **-0.5** (이전 -5e-6, ~100,000× 강화)
- 5번 적재 후 NOOP: ep_return = +4.46 (= 5 placements - 0.5)
- 정상 완주: ep_return = +125.98
- **NOOP 선택 시 총 회피 신호 ≈ 126** (= 0.5 명시 + 125.5 기회비용)

**검증된 정합성**:
- `sum(_ep_reward_terms.values()) == _ep_return` 정합 유지(7-2 보존).
- `terms["fail"] = penalty`로 단일 키 누적 → TB의 `train/reward_ep/fail`이 의미 있는 음수 값으로 노출됨(이전엔 ~0 묻힘).

**효과**:
- NOOP/early-stop 억제 신호가 dense하게 들어옴 → 초기 exploration이 NOOP만 골라 stuck되는 함정 방지.
- 물리 fail은 여전히 가벼워 masking 신뢰 정책 유지.
- 옵션 D의 sparse terminal_su를 보완하는 explicit immediate penalty 역할.

### 7-5. r_dead 컨셉 재정렬 — margin-bridge 자연 상쇄 + DEAD_WEIGHT 격하 (2026-05-02)

**배경 — 사용자와의 컨셉 align**:
- 사용자 원본 의도(이전 코드 `dead_increase > 1e-4` 임계값):
  ① "이 step에서 새로 생긴 dead만 처벌, 과거 dead는 재처벌 안 함" — RL credit assignment 원칙.
  ② "margin 위에 아이템을 덮어 생긴 dead는 면제" — margin-bridge는 정의상 필연적 + transport 안정성에 기여.
- 옵션 D 리팩토링(state-based 전환)에서 임계값이 사라지고 `r_dead = -DEAD_WEIGHT × dead_ratio × 10`만 남음 → ①은 state-based delta가 자동 보장하지만 ②는 보호장치 소실.
- 게다가 이전 임계값(`> 1e-4` 절대값)은 사실상 numerical noise filter 수준이라 ② 의도를 거의 못 잡았음 (margin-bridge로 생긴 큰 dead와 bad placement dead 모두 통과).

**컨셉 정정 (이전 docs ⚠️ 표시 정정)**:
- "delta 방식: 한 번 만든 dead volume은 이후 step에서 다시 벌점을 받지 않음" → 이건 비판이 아니라 **올바른 설계**. credit assignment 원칙과 일치.
- 사용자가 원래 임계값을 둔 이유(margin-bridge 면제) 컨셉은 옳음.

**선택된 해결책 — 자연 상쇄 (옵션 ②)**:
- 별도 geometry 검출 코드 없이, 이미 활성화된 placement-specific 보너스가 같은 일을 하고 있음:
  - margin-bridge: 두 지지 위에 걸침 → support_ratio≈1.0 → r_stability≈0, 두 아이템 접촉 → r_contact 큼.
  - bad floating: 한 모서리 살짝 → support_ratio≈0.6 → r_stability=-0.2, contact 작음.
- r_dead의 비중을 낮추고(DEAD_WEIGHT 1.0 → 0.3) "약한 가이드" 역할만 부여 → 진짜 placement 평가는 stab/contact가 담당.
- 사용자 결정 근거: "agent가 여러 경험을 하고 깨달아야 하는 거 같아. 나는 신호만 줘야 하고."

**수정 (`reward_builder.py`)**:
```python
# was DEAD_WEIGHT = 1.0 → 0.3
DEAD_WEIGHT: float = 0.3
```
한 줄 변경.

**비교 (1m³ bin, placement 유형별 추정 step 보상)**:

| placement 유형 | r_stability | r_contact | r_dead (delta) | net |
|---|---|---|---|---|
| Margin-bridge (지지 2개) | ~0 (완벽) | ~+5 (큰 접촉) | ~-0.05 | **~+4.95** |
| Tight (dead 0) | ~0 | ~+5 | 0 | **+5.00** |
| Bad floating (지지 약함) | -0.2 | ~+1 | ~-0.2 | **+0.6** |

→ margin-bridge가 페널티를 받지만 stab/contact 보너스가 압도적으로 커서 net + 보상. Tight placement는 여전히 미세하게 우위. Bad floating은 강하게 처벌. 컨셉과 일치.

**검증 (TSG 32-item, 정상 완주)**:
- ep_return = +125.98 (DEAD_WEIGHT 변경 전과 동일 — TSG 균일 시나리오라 dead=0).
- per-term 비중: qual_contact 47.6%, terminal_su 39.7%, alive 12.7%, bal ~0%, eff_dead 0%.
- mixed-size 시나리오에서 r_dead 효과는 학습 결과 관찰 필요(시급 트리거: TB의 `train/reward_ep/eff_dead` 절댓값이 stab/contact 합보다 클 때).

**미해결 (장기 백업안)**:
- 학습에서 margin-bridge가 여전히 회피되는 패턴 관찰 시 → 옵션 ①(geometry-aware bridging detection) 도입 검토.
- 그땐 `bin.render()` 시각화로 placement 패턴 디버깅 후 결정.

**효과**:
- 사용자의 원본 두 컨셉(① 현재 step만 평가, ② margin-bridge 면제) 모두 코드 의도대로 작동.
- geometry detection 코드 없이 단순한 가중치 조정으로 컨셉 align 달성.
- ep_return 절대 magnitude는 거의 변동 없음(SU=1.0이면 dead=0이라) → 학습 안정성 영향 적음. 변화는 실패-가까운 상태(낮은 SU, 높은 dead)에서 선명.

### 7-6. r_stability 지원 함수 캐싱 — `Item.update_face_cache` / `getVertices` (2026-05-02)

**배경 — 사용자 컨셉 결정 (옵션 A3)**:
- r_stability의 기하학적 측정 자체는 그대로 유지("물리적으로만 판단, 선택은 agent가").
- 다만 `_calculate_support_ratio_geometric` 호출 비용이 우려됨 → 컨셉 변경 없이 연산만 효율화.

**문제 진단 (cProfile)**:
- `_calculate_support_ratio_geometric` 한 번 호출 시 내부에서:
  ① `bin.get_bottom_items(item)` → `item.update_face_cache()` 강제 호출
  ② `update_face_cache` → `getVertices()` 호출 (numpy array 생성, rotation matrix 곱, np.isclose × 2)
  ③ 6개 면 전부 dict 재구성
- placed_item의 위치는 placement 시점에 fix되어 변하지 않는데도 매번 처음부터 재계산.
- 같은 함수가 다른 hot path(`is_top_empty`, `get_direction_overlap`, `checkPivot` 등)에서도 반복 호출됨 → cascade 비효율.

**수정 (`item.py`)**:

`getDimension` 캐싱(7-1과 같은 사례)과 동일 패턴으로 두 헬퍼에 cache key 추가.

```python
# __init__
self._verts_cache_key = None
self._verts_cache_val = None        # tuple of tuples — immutable
self._face_info_cache_key = None    # _face_info dict 자체가 캐시 저장소

def getVertices(self):
    bp, rq = self.b_position, self.rotation_quat
    cache_key = (bp[0], bp[1], bp[2], rq[0], rq[1], rq[2], rq[3],
                 self.width, self.height, self.depth)
    if self._verts_cache_key == cache_key and self._verts_cache_val is not None:
        return [list(v) for v in self._verts_cache_val]   # caller mutation 안전
    # ... 기존 numpy/rotation 로직 ...
    self._verts_cache_key = cache_key
    self._verts_cache_val = tuple(tuple(v) for v in verts)
    return verts

def update_face_cache(self):
    bp, rq = self.b_position, self.rotation_quat
    cache_key = (bp[0], bp[1], bp[2], rq[0], rq[1], rq[2], rq[3],
                 self.width, self.height, self.depth)
    if self._face_info_cache_key == cache_key and self._face_info:
        return   # cache hit — _face_info 그대로 사용
    # ... 기존 6면 빌드 로직 ...
    self._face_info_cache_key = cache_key
```

추가로 `__setstate__`에 backward-compat guard 추가(옛 pickle 로드 시 새 필드 None으로 초기화).

**불변성 보장**:
- 외부에서 `it.b_position = ...`, `it.rotation_quat = ...`, `it.width = ...` 등 어느 필드 변경 시 cache_key 자동 mismatch → 다음 호출에서 재계산.
- `getVertices` 반환은 매번 새 list — caller가 수정해도 캐시 안전.
- `_face_info` dict는 _그대로_ 재사용되지만 외부 mutation 사례 없음(grep 확인).

**검증 (정확성)**:
- cache hit 시 동일값 반환 ✓
- caller mutation 후 다음 호출 안전(반환 list 수정해도 다음 호출 결과 변함 없음) ✓
- `b_position`/`rotation_quat` 변경 시 invalidate ✓
- 전체 env step pipeline (33 step 완주 → SU=1.0 → ep_return=+125.98) 변동 없음 ✓

**성능 측정**:

| 항목 | 이전 (cache miss) | 현재 (cache hit) | 배율 |
|---|---|---|---|
| `getVertices()` | 64.3 μs/call | **0.97 μs/call** | ~67× |
| `update_face_cache()` | (포함됨) | **0.29 μs/call** | — |
| `_calculate_support_ratio_geometric()` | ~50-100 μs/call (추정) | **0.9 μs/call** | ~50-100× |
| `build_reward()` 전체 | ~1ms (추정) | **0.64 ms/call** | ~30-40% |
| env step (TSG, 평균) | ~155 ms/step | **~115 ms/step** | ~25-30% |

**Cascade 효과 확인 (build_reward 프로파일, 200 calls)**:
- `_calc_support_ratio`: 사실상 사라짐 (0.2% 비중)
- 새 hot spot:
  - `_get_contact_score / get_direction_overlap`: 55% (다음 cache 후보)
  - `_dead_ratio / get_AFV`: 26%
  - `balance_term_capped`: 14%

**호출처 cascade 효과**:
- `get_bottom_items` (`_calculate_support_ratio_geometric`)
- `is_top_empty`
- `get_direction_overlap` (이웃 아이템들의 `getVertices` 반복 호출)
- `checkPivot` (지지/충돌 검증, 자식 아이템들 face cache 빈번 호출)
- `pivot_generation` (모든 아이템 vertices 수집)

**효과**:
- r_stability 함수 호출 자체는 무시할 수 있는 비용으로 떨어짐.
- placement-specific 보너스 계산 + checkPivot 등 다른 hot path 동시 가속.
- 컨셉 변경 없이 순수 연산 효율 개선만으로 step time 25-30% 단축.

**미해결 (다음 cache 후보)**:
- `get_direction_overlap` 내부의 R-tree 검색 결과: bin.size 기준 캐싱 가능. step당 한 번만 새로 계산하면 충분.
- `_dead_ratio / get_AFV`: bin.size 변화에 invalidate되는 캐시 가능.
- 이건 build_reward 후속 최적화로 별도 검토.

### 7-7. r_contact 스케일 정규화 + CONTACT_WEIGHT 0.1 → 1.0 (2026-05-02)

**문제 진단**:
- 이전 식: `r_contact = 0.1 × (contact_mm² / 10000)` — `/10000`은 임의 하드코딩(100×100mm 한 면을 1.0으로 보는 기준).
- bin/박스 크기에 비례 → 같은 "꽉 끼게 놓음" 행동이 시나리오별로 5~50× 다른 보상.
- 상한 없음 → 큰 박스 + 4면 접촉이면 r_contact +5 ~ +20 가능.
- 디버그 트레이스(TSG 32 items, 정상 완주):
  - `qual_contact` 누적: **+60** (전체 ep_return의 47.6%, 1위)
  - `terminal_su`: +50 (39.7%, 2위)
  - **contact가 옵션 D의 진짜 목표 신호(terminal_su)를 압도** → 정책이 SU보다 contact를 먼저 추구할 위험.

**연산 효율 검토**:
- `_get_contact_score` → `get_direction_overlap` → 4 방향 × `direction_gap_rtree`.
- 방향당 1× `search_xyz` (3 R-tree 교집합) + 후보 정렬 + per-candidate 산수.
- 이미 7-1 (`getDimension` 캐시), 7-6 (`update_face_cache`/`getVertices` 캐시) cascade로 한 번 빨라진 상태.
- 200 calls 측정: ~70ms = **0.35ms/call**. 24 envs × 30 step × 0.35ms = ~250ms/rollout. **무시할 수준**.
- 추가 캐시 후보(R-tree 결과 캐싱 등)는 학습 결과 본 뒤 결정. 이번엔 **스케일 정규화 위주**.

**수정 (`reward_builder.py`)**:

```python
# was CONTACT_WEIGHT = 0.1 → 1.0 (정규화 후 스케일 보정)
CONTACT_WEIGHT: float = 1.0

def _item_lateral_area(item) -> float:
    """item의 4개 측면 면적 합 (= 측면 접촉의 이론적 최대)."""
    w, h, d = item.getDimension()
    return 2.0 * d * (w + h)

# build_reward 내부
contact_val = _get_contact_score(bin_obj, placed_item)
max_lateral = _item_lateral_area(placed_item)
r_contact = CONTACT_WEIGHT * (contact_val / max(1e-6, max_lateral))   # ∈ [0, 1.0]
```

**정규화 근거 — `2 × D × (W + H)`**:
- left + right 측면 면적: 2 × (H × D)
- front + back 측면 면적: 2 × (W × D)
- 합: 2D(H+W) = 측면 4개 전부 가득 접촉 시의 이론적 최대.
- contact_val / max_lateral = "측면 둘레 중 접촉 비율" ∈ [0, 1].

**검증**:

| placement (1m³ bin) | contact_val | max_lateral | r_contact (이전) | r_contact (현재) |
|---|---|---|---|---|
| 100³ 박스 한 면 접촉 | 10,000 | 40,000 | 0.1 | **0.25** |
| 250³ 박스 한 면 접촉 | 62,500 | 250,000 | 0.625 | **0.25** |
| 500³ 박스 한 면 접촉 | 250,000 | 1,000,000 | 2.5 | **0.25** |
| 250³ 박스 두 면 접촉 | 125,000 | 250,000 | 1.25 | **0.50** |
| 250³ 박스 4면 접촉 | 250,000 | 250,000 | 2.5 | **1.00** (상한) |

→ 같은 "한 면 접촉" 행동은 박스 크기 무관하게 동일 보상(0.25). bin/박스 크기 편향 소멸.

**전체 신호 hierarchy 회복 (TSG 32 items 정상 완주)**:

| term | 이전 | 현재 | 비중 변화 |
|---|---|---|---|
| `terminal_su` | +50 (39.7%, 2위) | +50 (**64.1%, 1위**) | ↑↑ (옵션 D 의도 회복) |
| `alive` | +16 (12.7%, 3위) | +16 (20.5%, 2위) | — |
| `qual_contact` | +60 (**47.6%, 1위**) | +12 (15.4%, 3위) | ↓↓ (압도 해소) |
| `bal` | -0.02 | -0.02 | — |
| **ep_return** | +125.98 | **+77.98** | -38% |

step별 contact 분포:
- 이전: +1.25 ~ +5.0 (mean ~2.0)
- 현재: +0.25 ~ +1.00 (mean 0.52) → alive(+0.5)와 비슷한 스케일

**효과**:
- ✅ 옵션 D의 신호 hierarchy 회복: terminal_su가 압도적 1위(64%).
- ✅ bin/박스 크기 무관한 보상 → 시나리오별 일관성.
- ✅ step당 [0, 1.0] 상한 → critic 학습 분산 ↓.
- ⚠️ ep_return 절대값 38% 감소 → PPO learning rate 영향 없음(advantage normalization 사용). TB의 `train/reward_ep/qual_contact` 절댓값이 전후로 5× 감소 보일 것.

**미해결 (다음 build_reward hot spot)**:
- `_dead_ratio / get_AFV` 26% 비중 (build_reward 안에서). bin.size 변화에만 invalidate되는 캐시 가능.
- `balance_term_capped` 14%. 마찬가지.
- 측정값 기준 학습 곡선이 만족스러우면 추가 작업 불필요.

### 7-8. r_bal terminal로 이전 — `balance_term_capped` 제거 + `terminal_bal` 도입 (2026-05-02)

**문제 진단**:
- 이전: `r_bal = BALANCE_WEIGHT × balance_term_capped(1.0, bin, ...)` per-step.
  - magnitude: ±0.025/step, 32-item 누적 -0.016 (사실상 0%, 학습 신호 무영향).
- `balance_term_capped` 내부의 cap 식 `cap = 0.1 × |su_term|`은 옵션 D(SU_WEIGHT=0)와 충돌 → 호출자가 sloppy fix로 `1.0` 고정 전달 → cap 의도 흐려짐.
- 컨셉: "transport 균형"은 매 step 변화가 아니라 **적재 완료 시점의 결과** → terminal 일괄 지급이 의미적으로 정확.

**함수 정당성 점검 (사용자 요청)**:

`balance_feature` + `normalize_balance_features`는 수학적으로 타당:
- **`imbalance_norm`** = ||Σ mₜ × (rx, ry)|| / (√0.5 × Σmₜ) — COM 쏠림. √0.5는 \|(0.5, 0.5)\| = 이론적 최대.
- **`I_norm`** = Σ 0.5 × mₜ × (rx²+ry²) / (0.25 × Σmₜ) — 분포 응집도. 0.25 = 0.5×(0.5²+0.5²) = 이론적 최대.
- 두 신호는 redundant 아님: 좌우 대칭이지만 끝에 몰려 있으면 imbalance=0이지만 I 큼 → 다른 정보 잡음.
- 평균 score = 0.5 × ((1-imb_n) + (1-I_n)) ∈ [0, 1] (1=완벽 균형) — 의미 명확.
- 검증 (분리 시나리오):
  - 대각선 두 모서리 (대칭, 응집X): imb_n=0, I_n=0.64 → score=0.68
  - 중앙 응집:                    imb_n=0, I_n=0    → score=1.00
  - 직관과 일치 ✓

→ **수식은 그대로 두고 사용 위치만 per-step → terminal로 이전**.

**선택된 해결책 — 옵션 ② (terminal로 이전)**:
- per-step `r_bal` 제거 → 옵션 D 철학(진짜 목표는 episode 끝 평가)과 일관.
- `get_terminal_bonus`가 final balance score를 추가로 계산해 `terminal_bal`로 지급.
- magnitude 의미있게 ([0, 5]) → terminal_su(50)의 1/10 비중, transport 신호로 적정.

**수정 (`reward_builder.py`)**:

```python
# 가중치
TERMINAL_BAL_WEIGHT: float = 0.5    # NEW (was BALANCE_WEIGHT 0.5 per-step)
# BALANCE_WEIGHT 상수 + balance_term_capped import 제거
# balance_feature, normalize_balance_features를 직접 import해서 terminal에서만 사용

# build_reward — r_bal 제거
def build_reward(bin_obj, placed_item=None):
    r_su = SU_WEIGHT * bin.SU * 10           # SU_WEIGHT=0
    r_dead = -DEAD_WEIGHT * dead_ratio * 10
    # r_bal 제거 (terminal로 이전)
    r_alive, r_stab, r_contact = ...
    total = r_su + r_dead + r_alive + r_stab + r_contact
    terms = {"eff_su":..., "eff_dead":..., "alive":..., "stab_soft":..., "qual_contact":...}
    return total, terms

# get_terminal_bonus — terminal_bal 추가
def get_terminal_bonus(bin_obj):
    su_bonus = TERMINAL_SU_WEIGHT * bin.SU * 10
    imb_raw, I_raw, sum_m_t = balance_feature(bin_obj)
    imb_n, I_n = normalize_balance_features(imb_raw, I_raw, sum_m_t)
    bal_score = 0.5 * ((1.0 - imb_n) + (1.0 - I_n))   # ∈ [0, 1]
    bal_bonus = TERMINAL_BAL_WEIGHT * bal_score * 10  # ∈ [0, 5]
    return su_bonus + bal_bonus, {
        "terminal_su":  su_bonus,
        "terminal_bal": bal_bonus,
    }

# _TERM_KEYS 갱신
_TERM_KEYS = ("eff_su", "eff_dead", "alive", "stab_soft", "qual_contact",
              "terminal_su", "terminal_bal")    # "bal" 제거, "terminal_bal" 추가
```

env.py는 변경 불필요: `_finalize_step`이 이미 `t_terms`의 모든 키를 `delta_terms`에 합쳐 누적함.

**검증 (TSG 32 items 정상 완주)**:

| 항목 | 이전 (per-step bal) | 현재 (terminal_bal) |
|---|---|---|
| ep_return | +77.98 | **+82.22** (+4.24, terminal_bal 신규) |
| 정합 (sum_terms == ep_return) | True | **True** ✓ |
| `terminal_su` | +50.0 (64.1%) | +50.0 (60.8%) |
| `alive` | +16.0 (20.5%) | +16.0 (19.5%) |
| `qual_contact` | +12.0 (15.4%) | +12.0 (14.6%) |
| `terminal_bal` | — | **+4.22 (5.1%)** ← NEW |
| `bal` (per-step) | -0.016 (0.0%) | 제거 |
| step별 bal magnitude | ±0.025 | 0 (per-step 없음) |

**효과**:
- ✅ "Final balance"로 의미 정확화: 매 step 변화가 아니라 적재 완료 시점의 transport 안정성.
- ✅ Magnitude 의미있게 ([0, 5]) — 이전 0% 비중 → 5.1% 비중. 학습 신호로 작동 가능.
- ✅ 옵션 D 철학 일관: SU + bal 모두 terminal에서 일괄 지급. dense 신호는 alive + stab + contact가 담당.
- ✅ 코드 정리: `balance_term_capped`(SU=0과 충돌하던 sloppy cap 식) 의존성 제거. `BALANCE_WEIGHT` 상수 → `TERMINAL_BAL_WEIGHT`로 교체(이름이 의미 명확).
- ✅ 연산: per-step balance_feature 호출(prev + curr 2회) → episode당 1회로 감소 (~30 calls/episode → 1 call/episode).

**행동 인센티브 변화**:
- 이전: balance 신호 거의 무시 → 정책이 균형 고려 안 함.
- 현재: terminal_bal +4.22 (full balance 시) vs 0 (한쪽 쏠릴 때) → episode 끝에서 적재 분포에 명시적 보상. NOOP/early-stop 시 미지급 → terminal_su(50) + terminal_bal(5)= 55점 기회비용.

**남은 작업**: roadmap 즉시 수정 항목 모두 적용 완료. ⑦ r_flat은 별도 검토 후 영구 삭제 (Sec 7-9).

### 7-9. r_flat 영구 삭제 (2026-05-02)

**배경**:
- 이전 상태: 코드는 존재하지만 build_reward에서 호출 안 됨(주석 처리). `FLATNESS_WEIGHT=0.002`, `_get_surface_std` 헬퍼, `collect_ez_stats` import만 잔존.
- 사용자와 컨셉 토론 후 **부활 시도 대신 영구 삭제** 결정.

**삭제 근거 (6가지)**:

1. **obs와 정보 중복**:
   - `globals.h_score`(observation)가 이미 height distribution 점수를 policy에 노출.
   - 보상 신호로서 r_flat의 추가 정보량 ≈ 0.

2. **다른 reward 신호와 자연 상관**:
   - 평탄하게 쌓으면 SU↑, dead↓ → terminal_su가 implicit 보상.
   - r_stability/r_contact도 좋은 placement에 양의 신호.
   - r_flat 고유 기여분이 작음.

3. **옵션 D 철학과 충돌**:
   - "최종 목표(final SU + bal)만 직접 평가, 나머지는 dense gradient용 최소 신호"가 옵션 D 원칙.
   - r_flat은 forward-looking proxy 신호 → 추가하면 minimalism 역행.
   - 사용자 본인의 표명("agent가 여러 경험을 하고 깨달아야 하는 거 같아. 나는 신호만 줘야 하고")과 일치.

4. **Forward-looking은 critic의 일**:
   - "이 상태가 미래에 좋다"는 PPO의 value function이 학습할 영역.
   - 진짜 목표(terminal_su)에서 backprop되는 advantage가 자연스럽게 평탄한 상태를 가치 있다고 평가.
   - reward에 박으면 critic 학습 의미 왜곡.

5. **Misdirection 위험**:
   - 작은 박스를 굳이 위에 올려 평탄도 맞추는 행동 학습 가능.
   - 사실 그 박스를 floor에 놓는 게 final SU에 더 좋을 수 있음 → 진짜 목표와 충돌.
   - cap 없으면 std 차이가 박스 크기에 비례해서 ΔV 편향 재발 가능.

6. **이미 비활성 + 무영향 검증됨**:
   - FLATNESS_WEIGHT=0.002로 거의 0 가중. 이전 실험에서 제거해도 학습 차이 없었다고 사용자 판단.

**수정 (`reward_builder.py`)** — 4곳 정리:
- `FLATNESS_WEIGHT` 상수 삭제
- `_get_surface_std(bin_obj)` 헬퍼 함수 삭제
- `collect_ez_stats` import 삭제 (다른 호출처 없음 확인)
- `import numpy as np` 삭제 (다른 호출처 없음 확인)

**검증**:
- `hasattr(reward_builder, 'FLATNESS_WEIGHT')` = False ✓
- `hasattr(reward_builder, '_get_surface_std')` = False ✓
- TSG 32 items 정상 완주: ep_return = +82.22 (변동 없음, r_flat은 원래 호출 안 되던 코드)
- `sum(_ep_reward_terms) == ep_return` 정합 유지 ✓

**재도입 트리거 (장기)**:
학습 결과 관찰에서 다음 패턴 보이면 재고:
- eval에서 SU=1.0이지만 윗면 들쭉날쭉 → 다음 batch 적재 어려운 시나리오.
- 정책이 "큰 박스 한쪽 쌓기"만 고집하고 평탄도 무시.

**재도입 시 설계 가이드**:
- per-step `r_flat`은 **금지** (옵션 D 철학 충돌).
- terminal로만: `terminal_flat = TERMINAL_FLAT_WEIGHT × (1 - std/max_std) × 10`, magnitude 작게 ([0, 2] 정도).

**효과**:
- 코드 단순화 (4 곳 정리).
- 옵션 D 철학 일관성 강화: per-step proxy 신호 흔적 완전 제거.
- 학습 신호 변동 없음 (이전부터 비활성이었으므로).
