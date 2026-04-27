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
| `SU_WEIGHT` | 1.6 | 공간 효율 gain |
| `DEAD_WEIGHT` | 1.0 | 데드 볼륨 증가 페널티 |
| `STABILITY_PENALTY_WEIGHT` | 0.5 | 지지율 부족 soft penalty |
| `FLATNESS_WEIGHT` | 0.002 | 윗면 균일도 (미사용) |
| `CONTACT_WEIGHT` | 0.1 | 접촉 면적 보너스 |
| `BALANCE_WEIGHT` | 0.5 | 무게중심 균형 |

### 1-3. PENALTY_MAP

| failure_code | 페널티 (설정값) | 실제값 (×0.001) |
|---|---|---|
| `FAIL_COLLISION` | -5.0 | **-0.005** |
| `FAIL_OUT_OF_BOUNDS_*` | -5.0 | -0.005 |
| `FAIL_NO_TOP_EMPTY` | -5.0 | -0.005 |
| `FAIL_NO_SUPPORT_BOTTOM` | -3.0 | -0.003 |
| `FAIL_CG_OUTSIDE_SUPPORT` | -3.0 | -0.003 |
| `FAIL_CUMULATIVE_UNSTABLE` | -3.0 | -0.003 |
| `FAIL_WEIGHT_EXCEEDED` | -3.0 | -0.003 |
| `FAIL_SUPPORT_OVERLOAD` | -3.0 | -0.003 |
| `FAIL_OVERHANG_TOO_MUCH` | -2.0 | -0.002 |
| `FAIL_SUPPORT_AREA_INSUFFICIENT` | -2.0 | -0.002 |
| "default" | -5.0 | -0.005 |

---

## 2. 항목별 상세 및 타당성

### A. `r_su` — 공간 효율 gain ⭐ (핵심)

```python
su_gain = (after_bin.SU - before_bin.SU) * 10.0
r_su = SU_WEIGHT * su_gain     # 1.6 × ΔSU × 10
```

**설계 의도**: 공간 활용률(Space Utilization) 증가분을 직접 보상.

**타당성**:
- ✅ binpacking의 핵심 목표인 SU를 직접 최적화 → PPO가 가치 함수를 학습하기 가장 좋은 신호.
- ✅ delta 방식이라 에피소드 길이에 의존하지 않음 (누적이 항상 최종 SU와 일치).
- ⚠️ 큰 아이템과 작은 아이템의 ΔSU 차이가 수십 배 → 큰 아이템 초반에 배치하면 보상이 튐.
  (normalized 후 × 10이라 대략 [0, 수]의 범위)

**평가**: 현재 구조의 **주 보상**. 잘 동작하고 있을 것으로 추정.

---

### B. `r_dead` — 데드 볼륨 페널티

```python
dead_increase = _dead_ratio(after) - _dead_ratio(before)
r_dead = 0.0
if dead_increase > 1e-4:
    r_dead = -DEAD_WEIGHT * dead_increase * 10.0
```

**설계 의도**: 아이템을 쌓을 때 주변에 **갇혀서 쓸 수 없게 된 공간**이 생기면 벌점.

**타당성**:
- ✅ 공중 부양/구석 배치처럼 SU는 올리지만 빈틈을 만드는 배치를 억제.
- ⚠️ `dead_increase > 1e-4`인 경우에만 페널티 → **작은 증가가 여러 번 누적되면 놓침**.
- ⚠️ delta 방식: 한 번 만든 dead volume은 이후 step에서 다시 벌점을 받지 않음.

**평가**: 개념은 맞지만 임계값 기반이라 **누적 dead volume**에 대한 압력이 부족.

---

### C. `r_stability` — 지지율 soft penalty ⚠️ **현재 항상 0**

```python
r_stability = 0.0
if cand_item is not None:
    support_ratio = _calculate_support_ratio_geometric(after_bin, cand_item)
    r_stability = (support_ratio - 1.0) * 0.5
```

**설계 의도**:
- Action Masking이 **최소 지지율**(예: 0.6)을 이미 보장한 상태.
- 여기서는 **"얼마나 완벽히(1.0)에 가까운가"**를 추가로 보상.
- 지지율=1.0 → 0점 (완벽), 지지율=0.6 → -0.2점 (감점).

**타당성**:
- ✅ Soft constraint 설계 철학이 명확하고 합리적.
- ✅ checkPivot이 최소 기준은 걸러주므로 reward는 fine-tuning에만 관여.
- ❌ **치명적 문제**: `env.py`의 `_finalize_step`이 `cand_item`을 전달하지 않아 **항상 0**.

**평가**: **설계 O, 동작 X**. env 수정으로 `placed_item` 전달 시 즉시 활성화됨.

---

### D. `r_contact` — 접촉 면적 보너스 ⚠️ **현재 항상 0**

```python
contact_val = _get_contact_score(after_bin, cand_item) if cand_item else 0.0
r_contact = CONTACT_WEIGHT * (contact_val / 10000.0)
```

**설계 의도**: 주변 아이템/벽과 접촉 면적이 클수록 보너스 → 밀착 배치 유도.

**타당성**:
- ✅ 접촉 면적 ↑ = 안정성 ↑ + dead volume ↓ → 간접적 품질 지표.
- ⚠️ `/10000.0` 스케일이 **하드코딩**. bin 크기나 아이템 크기에 따라 단위 비용 변동.
  - mm² 단위라고 가정: 100mm × 100mm = 10000 (1개 면적 기준)
  - 하지만 실제 아이템은 훨씬 큼 (300~800mm) → contact_val이 수만~수십만 나올 수 있음.
- ❌ **cand_item=None이라 항상 0**.

**평가**: **설계 O, 동작 X + 스케일 검증 필요**. 활성화 시 bin 크기 대비 정규화하는 게 안전.

---

### E. `r_bal` — 무게중심 균형

```python
bal_term_val = balance_term_capped(r_su, after_bin, cand_item=None)
r_bal = BALANCE_WEIGHT * bal_term_val
```

`balance_term_capped`:
- 현재 bin의 불균형(`imb_n`)과 관성(`I_n`)을 계산 → `score ∈ [0,1]`
- 최종 출력: `cap × (2·score - 1) ∈ [-cap, +cap]`, `cap = 0.1 × |r_su|`

**설계 의도**:
- 무게 중심이 bin 중앙에 가까울수록 양수, 멀수록 음수.
- SU 보상의 10% 이내로 cap 걸어 무게 항이 SU를 압도하지 않게 함.

**타당성**:
- ✅ Capping으로 스케일 안정성 확보.
- ⚠️ `cand_item=None`이라 **전역 균형**만 계산 → 방금 놓은 아이템이 균형에 **기여한 incremental 효과**를 측정하지 않음.
  - 즉, 같은 step에서 어떤 위치에 놓았는지와 무관하게 bin 전체의 현재 균형 상태만 반영.
  - 실제로 `balance_feature(bin, cand_item=X)`로 호출하면 X를 포함한 가정 하의 균형을 계산 가능.
- ⚠️ `r_su = 0`이거나 작을 때 `cap`도 0에 가까워 → **학습 초반엔 거의 안 먹힘**.

**평가**: cap 로직은 OK, **cand_item 전달하면 정확도 상승**. 현재도 0은 아니고 약하게 작동 중.

---

### F. `r_flat` — 윗면 균일도 (주석 처리됨)

```python
# std_before = _get_surface_std(before_bin)
# std_after = _get_surface_std(after_bin)
# flatness_gain = std_before - std_after
# r_flat = FLATNESS_WEIGHT * flatness_gain
```

**설계 의도**: 쌓인 아이템들의 **윗면 높이 표준편차**를 측정, 감소 시 보너스 → 계단 대신 평평한 층을 만들도록 유도.

**타당성**:
- ✅ 다음 아이템을 놓을 플랫폼을 평평하게 유지 → 장기적으로 적재 가능 공간 확보.
- ⚠️ SU 보상과 **시그널이 중복/상충 가능**:
  - 평평하게 쌓으면 대부분 SU도 증가 → 중복 신호.
  - 단, 작은 아이템으로 큰 구멍을 메우는 행동을 유도하는 데는 flatness가 유일한 시그널.
- ⚠️ `_get_surface_std`는 `collect_ez_stats`에서 높이 분포를 수집 → **N개 아이템 순회 O(N)**.
  매 step 호출하면 비용 무시 못함.

**왜 주석 처리되었을까?**:
- 가중치(0.002)가 매우 작아 학습 신호 기여가 미미하다고 판단.
- SU/dead와 중복 → 제거해도 학습 성능 차이 안 보였을 가능성.
- 또는 observation의 `globals`에 `h_score`(height distribution)가 이미 포함 → policy에게 정보는 제공됨.

**평가**:
- 실험에서 성능 영향이 관측되지 않았다면 제거가 타당.
- 다만 **globals 인코딩에만 들어있고 reward 신호가 없으면, policy가 평탄화를 학습할 유인은 약화**된다.
- 살릴 거라면 **가중치 상향(0.002 → 0.01 ~ 0.02)** + step마다 계산 비용 고려.

---

### G. `ALIVE_BONUS` = 0.0

**설계 의도** (원래): 배치 성공 시 기본 보상 → "뭐라도 놓으면 +". 조기 NOOP을 억제하는 일반적 트릭.

**타당성**:
- 현재는 0이므로 **없는 것과 동일**.
- `r_su > 0`이 충분히 크다면 ALIVE_BONUS는 redundant.
- 단, **ΔSU가 매우 작은 아이템**(작은 박스)이 거의 보상을 못 받을 때 문제 → 작은 아이템을 무시하는 정책을 학습할 수 있음.

**평가**: 전략적 선택. 현재 0.0은 의도된 것으로 보이나, 작은 아이템 학습이 약하다면 재고.

---

### H. 실패 페널티 — `PENALTY_MAP × 0.001` ⚠️ **사실상 0**

```python
if terminated and not finished:
    penalty = PENALTY_MAP.get(failure_code, -5.0)
    ...
    # 가능한 경우의 수만 제공할때는 penalty역할 거의 없애기
    penalty *= 0.001
    return float(penalty), terms
```

**설계 의도 (원 주석)**: "가능한 경우의 수만 제공할 때는 penalty 역할 거의 없애기"
- Action Masking이 **이미 물리적으로 가능한 action만 마스크=1**로 남김.
- 따라서 masked action을 골라 실패할 확률이 낮다고 가정 → 페널티 약화.

**타당성**:
- ✅ **일리 있음**: check=True로 만든 mask가 견고하면, 실패는 edge case.
- ❌ **그러나 현실은**: `check=True`가 완벽하지 않고 (에러/retry 발생), 또 실패 원인에는 `NOOP 선택`, `max_steps 초과`, `retry 한계` 같은 "정책적 실패"도 포함됨.
- ❌ 성공 보상이 `r_su ≈ 0.1~0.5`일 때 실패 페널티 `-0.005`는 100배 차이 → **실패해도 거의 무손실**.
  → 정책이 모험하다 실패해도 비용을 거의 지불하지 않음.
- ❌ `ALIVE_BONUS=0`, `r_su`가 작을 때 → 실패와 NOOP 구별이 거의 없어짐.

**구체 시나리오**:
- 아이템 10개 중 6개 쌓고 NOOP → `no_op_selected`로 `"default"` penalty `-0.005` 적용.
- vs. 10개 다 쌓고 `no_items`로 성공 종료 → penalty 0.
- 차이는 -0.005뿐이라, 초반의 누적 r_su 이득이 훨씬 크면 NOOP이 편할 수 있음.

**평가**: ×0.001은 너무 공격적. **× 0.01~0.1 정도가 적정선**. 또는 실패 종류별 차등 적용 (NOOP은 강하게, 물리 실패는 약하게).

---

## 3. 실제로 비활성화된 항목 — 핵심 문제 요약

| 항목 | 설계값 | 실제 동작 | 원인 |
|---|---|---|---|
| `r_stability` | soft penalty [-0.2, 0] | **항상 0** | env에서 `cand_item` 미전달 |
| `r_contact` | 밀착 보너스 | **항상 0** | env에서 `cand_item` 미전달 |
| `r_bal` (incremental) | cand 기여분 반영 | **전역 bal만 측정** | `balance_feature(cand_item=None)` 고정 |
| `r_flat` | 평탄화 유도 | **주석 처리** | 사용자가 실험 후 무효화 판단 |
| 실패 페널티 | -5 ~ -2 | **-0.005 ~ -0.002** | `× 0.001` 공격적 감쇠 |
| `ALIVE_BONUS` | 기본 보너스 | **0** | 값 0 설정 |

**즉, 현재 학습은 사실상 `r_su + (약한) r_dead + (약한) r_bal` 세 항목으로 돌아가고 있음.**
설계 문서상 6개 항목이 있다고 기대되는 것과 큰 차이.

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

## 5. 권장 적용 순서

| 순서 | 항목 | 난이도 | 기대 효과 |
|---|---|---|---|
| 1 | **build_reward 시그니처: `before_bin` → `before_SU, before_dead`** | 쉬움 | env A 병행, deepcopy 제거 가능 |
| 2 | **`cand_item` 전달 (env B와 묶음)** | 쉬움 | stability/contact 활성화 |
| 3 | **실패 페널티 × 0.001 → × 0.05 (차등 scale)** | 매우 쉬움 | NOOP 남발 억제 |
| 4 | **`r_bal`에 `cand_item` 전달** | 매우 쉬움 | 배치 효과 반영 |
| 5 | **`r_contact` 스케일 정규화** | 쉬움 | bin 크기 무관한 보상 |
| 6 | **완성 보너스 추가** | 쉬움 | 에피소드 완주 유인 |
| 7 | `r_flat` / `ALIVE_BONUS` 재검토 | 실험 필요 | 상황 따라 |

> **1~4번은 env.py 수정과 동시에 하나의 커밋으로 묶는 게 자연스럽다** (시그니처 변경이 양쪽에 걸려있음).

---

## 6. 요약

```
현재 reward = r_su(핵심) + r_dead(약) + r_bal(약)
             [r_stability, r_contact, r_flat, ALIVE_BONUS, failure_penalty는 사실상 비활성]

→ 설계된 6+1 항목 중 3개만 실질 작동

핵심 fix (env와 함께):
  ① build_reward(before_SU, before_dead, ...) 로 시그니처 변경 (deepcopy 제거 준비)
  ② placed_item을 cand_item으로 전달 (stability/contact 활성화)
  ③ 실패 penalty × 0.001 → × 0.05 (NOOP 억제)
  ④ r_bal에 cand_item 전달 (incremental 측정)
  ⑤ contact 스케일 정규화 (bin 크기 무관)
```
