# PalletFit RL 성능 개선 방향

> 분석 대상: `agent.py`, `custom_value_policy.py`, `env.py`, `reward_builder.py`, `obs_builder.py`, `config.py`  
> 코드 수정 없음 — 방향성 정리 문서

---

## 목차

1. [Train Dataset](#1-train-dataset)
2. [RL Model (Architecture)](#2-rl-model-architecture)
3. [Environment (Env)](#3-environment-env)
4. [Reward](#4-reward)

---

## 1. Train Dataset

### 현재 구성

| 항목 | 내용 |
|---|---|
| 데이터 모드 | `offline` (고정 JSON) / `online_type` (타입 기반) / `tsg` (공간 분할 생성) |
| 커리큘럼 | 시간 기반 3단계: bootstrap → mid → late |
| Bootstrap | `OVERFIT_TARGET_PATH` 단일 파일, 시드 고정, preview_cnt=PREVIEW_MAX(10) |
| Mid | offline 50% + random 50% |
| Late | offline 40% + random 60% |
| Rehearsal | 이전에 본 offline 파일 재학습 비율: 10% → 15% → 20% |
| DatasetHistory | 소스별 SU EMA 추적 → "한 번도 안 본 파일" 우선 선택 |
| TSG | `init_slice=(4,4,2)` 고정, `tsg_ratio=0.0` (사실상 비활성화) |
| 평가 | 3 에피소드, 고정 캐시 플랜 재사용 |

### 현재 문제점

**A. `plan_maker_hybrid`의 잠재 버그**

`agent.py`의 `plan_maker_hybrid` 함수는 `self.cfg.selectable_len_choices`와 `self.cfg.preview_k_choices`를 참조하지만 `AgentConfig`에 존재하지 않음. `use_adaptive_phase=False`라 현재는 실행되지 않지만, adaptive phase를 켜면 바로 `AttributeError` 발생.

```python
# agent.py:1218 — 존재하지 않는 필드 참조
selectable_len_choices=self.cfg.selectable_len_choices,
preview_k_choices=self.cfg.preview_k_choices,
```

**B. 평가가 너무 적음**

에피소드 3개로 SU 평균을 계산 → 분산이 크고 "best model" 판정이 불안정.

**C. TSG 다양성 부족**

`init_slice`가 (4,4,2)로 고정 → 항상 비슷한 크기/형태의 아이템 생성. `min_item_mm=100`도 고정.

**D. DatasetHistory가 선택에만 작용, 난이도 제어에 미사용**

"안 본 파일 우선"은 구현되어 있으나, **SU가 낮게 유지되는 어려운 파일을 더 많이 보여주는 hard-negative mining**은 없음.

**E. Eval 플랜이 정적 캐시**

학습 전체 동안 동일한 3개 에피소드로 평가 → 과적합 탐지 불가.

### 개선 방향

| 우선순위 | 항목 | 설명 |
|---|---|---|
| 상 | Bug fix: `plan_maker_hybrid` | adaptive phase 경로 인자 정리 |
| 상 | 평가 에피소드 수 증가 | 3 → 10~20, eval 플랜 랜덤화 (매 평가마다 새 시드) |
| 중 | Hard-negative mining | `DatasetHistory.ema_su < threshold`인 소스에 가중치 부여 |
| 중 | TSG 다양성 강화 | `init_slice`를 `(2~6, 2~6, 1~4)` 범위로 무작위 샘플링 |
| 중 | TSG 재활성화 | `tsg_ratio=0.3` 정도로 설정, 형태 다양성 확보 |
| 하 | preview_cnt 점진적 증가 | bootstrap에서 낮은 값(1~3)으로 쉬운 문제부터 학습 후 점차 10까지 |
| 하 | Adaptive phase 활성화 | 성능 정체 감지 후 phase 전환 실험 |

---

## 2. RL Model (Architecture)

### 현재 구성

#### Feature Extractor (`CustomCombinedExtractor`, 총 384차원)

| 입력 | 형태 | 인코더 | 출력 |
|---|---|---|---|
| `items_topk` (bin에 쌓인 아이템 상태) | (64, 22) | DeepSet (mean pool) | 128 |
| `globals` (SU, 높이분포, 무게균형 등) | (6,) | 2-layer MLP | 128 |
| `preview_queue` (다음 아이템 미리보기) | (10, 4) | Conv1d(k=3) + GAP | 64 |
| `act_cands` (배치 후보 좌표) | (3600, 4) | per-candidate MLP + masked mean pool | 64 |

#### Policy Backbone (`FTTransformerBackbone`)

- 384차원 → Linear → (8토큰 × 256) reshape
- CLS 토큰 prepend → 총 9 토큰
- TransformerBlock × 6 (d=256, 8heads, GEGLU FFN, RMSNorm, RoPE, DropPath)
- CLS 출력 → `head_pi`(256, SiLU), `head_vf`(256, SiLU)

#### Action Scoring (`PointerPolicyFT.scorer`)

- `latent_pi` (B, 256)을 (B, 3600, 256)으로 expand
- `_last_cand_emb` (B, 3600, 128)와 concat → Linear(384→128) → ReLU → Linear(128→1)
- Logits (B, 3600) → nan 처리 → clamp(-50, 50) → masking

### 현재 문제점

**A. 후보 임베딩 계산 비용 (핵심 병목)**

매 forward마다 3600개 후보 전체에 대해 Linear(4→128→128) 연산 수행. 유효 action은 수십~수백 개에 불과함에도 3600개 전체를 계산.

**B. `_last_cand_emb` 캐시 패턴의 fragility**

`features_extractor._last_cand_emb`를 인스턴스 변수에 저장하고, `_get_action_dist_from_latent`에서 `getattr`로 꺼내는 방식은 batch 처리나 병렬 환경에서 race condition 가능성 있음.

**C. CLS 토큰만으로 3600개 후보 점수 계산**

FT backbone이 처리하는 8개 토큰은 feature extractor의 요약 벡터를 단순 reshape한 것 — 각 후보가 어떤 아이템과 관련 있는지 **cross-attention이 없음**. CLS 하나의 요약 벡터가 3600개 모두에 동일하게 적용됨.

**D. preview_queue 인코더 약함**

Conv1d + GAP는 10개 미리보기 아이템을 하나의 64차원 벡터로 압축 → 각 아이템의 정체성 손실. 3번째 아이템과 7번째 아이템을 구분할 방법 없음.

**E. DeepSet의 mean pool에서 정보 손실**

bin에 쌓인 64개 아이템이 mean pooling으로 합산 → 최근에 쌓인 아이템, 최상단 아이템 등 중요한 아이템과 덜 중요한 아이템이 동일 가중치로 처리됨.

**F. 토큰에 RoPE 적용이 의미 없을 수 있음**

RoPE는 자연어/시계열처럼 위치에 순서가 있을 때 유효. 8개 토큰은 feature vector를 임의로 쪼갠 것 → 위치 의미 없음.

**G. preview_queue 피처가 너무 빈약 (4차원)**

`(w, h, d, weight)` 4가지만 인코딩 → 아이템의 회전 가능 여부, 종횡비, bin 대비 부피 비율 등 없음.

### 개선 방향

| 우선순위 | 항목 | 설명 |
|---|---|---|
| 상 | 유효 후보만 임베딩 계산 | `act_mask`로 마스킹된 유효 인덱스만 골라 sparse 연산 |
| 상 | Queue item별 임베딩 유지 | preview_queue를 GAP으로 뭉개지 말고 (10, d) 형태로 유지 |
| 상 | Cross-attention scorer | `latent_pi`와 per-candidate embedding 사이 attention 계산 |
| 중 | 계층적 액션 분해 | (1) 큐에서 아이템 선택 (10개 중 하나) → (2) 해당 아이템의 배치 후보 선택 (360개 중 하나) |
| 중 | items_topk에 attention 가중치 | 최상단/최근 아이템에 높은 가중치를 주도록 masked attention pooling |
| 중 | preview_queue 피처 강화 | 회전 옵션 수, bin 대비 부피, 종횡비 추가 |
| 하 | RoPE → Learnable position emb | 토큰 위치가 의미 없으므로 RoPE 제거 또는 learnable embedding으로 교체 |
| 하 | 공유 item encoder | bin items와 queue items에 동일한 인코더 사용하여 파라미터 효율화 |

---

## 3. Environment (Env)

### 현재 구성

| 항목 | 내용 |
|---|---|
| 환경 수 | 학습 24개 (SubprocVecEnv/spawn), 평가 1개 (DummyVecEnv) |
| Action space | Discrete(3600), NOOP = 인덱스 3599 |
| 종료 조건 | `no_items` / `no_op_selected` / `all_actions_failed` / `max_retries_exceeded` / `max_steps_reached` |
| 실패 처리 | 실패한 action을 마스크에서 제거, -0.01 페널티, 최대 100회 재시도 |
| 후보 재계산 | 매 step마다 `rebuild_candidates(check=True)` (EDP 방식) |
| Bin 상태 복사 | `copy.deepcopy(self._bin)` — 매 step 실행 |
| max_steps | `EnvPlan`에서 200으로 설정하나 env 기본값은 100 (plan이 덮어씀) |

### 현재 문제점

**A. `copy.deepcopy(self._bin)` 매 step**

R-tree 인덱스와 아이템 전체를 포함한 bin 객체를 매 step deepcopy → 학습 속도 저하의 주요 원인 중 하나. `before_bin`은 reward 계산에만 쓰임 (ΔSU, Δdead).

**B. cand_item이 항상 None**

`_finalize_step`이 `build_reward`를 호출할 때 `cand_item=None` — stability reward와 contact reward가 실제로는 **한 번도 계산되지 않음**.

**C. 실패 step에서도 후보 재계산**

실패 시 `_last_mask[action] = False`로만 마스크를 업데이트하면 되는데, 현재 코드에서는 `_finalize_step` 내부에서 불필요하게 `_safe_rebuild_candidates`를 호출할 수 있음.

**D. 평가 환경이 1개**

평가 시 DummyVecEnv 1개 환경으로 순차 실행 → 평가가 느리고, 에피소드 수를 늘리기 어려움.

**E. 재시도 횟수 한계 (100회)가 너무 높음**

유효 action이 수백 개인 상황에서 100회 재시도 허용 → 에피소드가 무의미하게 길어질 수 있음.

**F. Queue 정보가 obs에 포함되지 않음**

step 실행 이후 남은 아이템 수, 총 아이템 수 등이 observation에 없어 에이전트가 "얼마나 남았는지" 모름.

### 개선 방향

| 우선순위 | 항목 | 설명 |
|---|---|---|
| 상 | `copy.deepcopy` 제거 | ΔSU, Δdead를 step 전후 숫자값으로 바로 계산하거나 경량 스냅샷 사용 |
| 상 | `cand_item` reward에 전달 | `place_by_action`의 반환값(`placed_item`)을 `_finalize_step`에 넘겨 stability/contact reward 활성화 |
| 중 | 평가 환경 수 증가 | DummyVecEnv 5개로 병렬 평가 |
| 중 | 재시도 한계 축소 | 학습 초기 30 → 후기 5~10으로 점진적 감소 (curriculum) |
| 중 | 남은 아이템 수 obs 추가 | `globals` 벡터에 `queue_len / initial_item_count` 추가 |
| 하 | 실패 step 후보 재계산 생략 | 실패 시 마스크만 갱신, 후보 재계산은 성공 step에서만 수행 |
| 하 | 동적 max_steps | 아이템 개수 기반으로 episode 길이 자동 조정 |

---

## 4. Reward

### 현재 구성

```
total_reward = ALIVE_BONUS(0)
             + r_su       = 1.6 × (ΔSU × 10)
             + r_dead     = -1.0 × (Δdead × 10)   [증가할 때만]
             + r_stability = (support_ratio - 1.0) × 0.5   ← 실제로 항상 0
             + r_contact  = 0.1 × (contact_val / 10000)    ← 실제로 항상 0
             + r_bal      = 0.5 × balance_term_capped(...)

실패 종료:  PENALTY_MAP[code] × 0.001   (사실상 -0.005 ~ -0.003)
```

> `r_flat` (평탄도) 항목은 코드에 존재하지만 **주석 처리**됨

### 현재 문제점

**A. stability/contact 보상이 사실상 비활성**

`_finalize_step`에서 `cand_item=None`으로 `build_reward`를 호출 → `r_stability`와 `r_contact` 모두 항상 0. 보상 설계와 코드가 불일치.

**B. 실패 페널티가 사실상 0**

`PENALTY_MAP` 값 (예: -5.0)에 0.001을 곱하여 실제 페널티는 -0.005. 성공 보상이 `SU × 1.6 × 10 ≈ 0.1~0.5` 수준임을 고려하면 실패 비용이 100배 이상 낮음 → 에이전트가 실패를 감수하고 greedy하게 행동할 유인이 없어지지 않음.

> 의도적으로 낮춘 것이라면 주석으로 명시 필요

**C. r_dead의 임계값 의존성**

`dead_increase > 1e-4`일 때만 패널티 → 작은 증가는 누적되어도 무시됨. 절대값 기반 dead volume이 일정 비율 초과 시에도 패널티를 주는 방식이 더 안정적.

**D. 완성 보너스 없음**

모든 아이템을 다 쌓은 경우 (`no_items` 종료) 에 추가 보너스 없음 → "10개 쌓고 NOOP"과 "10개 모두 쌓은 에피소드"의 차이가 없음.

**E. contact 스케일 하드코딩**

`contact_val / 10000.0` — bin 크기나 아이템 크기와 무관하게 고정 스케일. bin 크기 변경 시 보상 크기가 달라짐.

**F. balance 계산에 cand_item=None**

`balance_term_capped(r_su, after_bin, cand_item=None)` → 방금 놓은 아이템의 무게 기여를 제외한 전체 무게 중심만 계산. 아이템 배치가 균형에 기여한 incremental 효과를 측정하지 않음.

### 개선 방향

| 우선순위 | 항목 | 설명 |
|---|---|---|
| 상 | cand_item 전달 fix | `env.py` 수정으로 `placed_item`을 reward에 넘겨 stability/contact 활성화 |
| 상 | 완성 보너스 추가 | `terminal_reason == "no_items"` 시 `+SU_final × k` 보너스 (k≈2~5) |
| 중 | 실패 페널티 재검토 | 0.001 곱셈 제거 또는 학습 phase에 따라 점진적으로 증가 |
| 중 | r_flat 재활성화 | 표면 균일도 보상을 다시 켜고 weight 조정 (FLATNESS_WEIGHT ≈ 0.01) |
| 중 | contact 스케일 정규화 | `contact_val / (bin_W × bin_D)` 형태로 bin 크기 대비 비율 사용 |
| 중 | r_dead 절대값 항 추가 | 현재 Δ 기반에 `dead_ratio_total × w` 추가하여 누적 낭비 공간 패널티 |
| 하 | balance incremental 계산 | before/after balance 차이로 계산하여 배치 기여분만 보상 |
| 하 | 보상 항목별 로깅 검증 | TensorBoard에서 각 항 (r_su, r_dead, r_stability 등) 값이 실제로 올바른지 확인 |

---

## 요약 우선순위

| 갈래 | 즉시 수정 가능 (버그/비활성) | 중기 개선 | 장기 연구 |
|---|---|---|---|
| Dataset | `plan_maker_hybrid` 인자 bug fix | 평가 에피소드 증가, hard-negative mining | Adaptive curriculum |
| Model | `_last_cand_emb` 패턴 정리 | 유효 후보만 임베딩, queue 인코더 강화 | 계층적 액션 분해, cross-attention scorer |
| Env | `cand_item` reward 전달 fix | deepcopy 제거, 평가 환경 수 증가 | 동적 max_steps |
| Reward | stability/contact 활성화 fix | 완성 보너스, 실패 페널티 재검토 | Potential-based shaping |

> **즉시 수정 3종**(Dataset bug, `cand_item` 전달, stability 활성화)은 코드 5~10줄 수정으로 현재 설계 의도대로 동작하게 만드는 것들. 먼저 이것부터 적용하고 성능 변화를 확인한 뒤 중기 개선으로 넘어가는 것을 권장.
