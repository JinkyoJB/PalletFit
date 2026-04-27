# Troubleshooting: RL Distribution / Simplex 검증 오류

## 개요

PalletFit-RL 추론 과정에서 `MaskablePPO.predict()` 호출 시 두 번째 스텝부터 발생하는 크리티컬 에러.  
아이템 1개가 성공적으로 적재된 직후의 스텝부터 crash가 발생하며, bin-packing RL 루프 전체가 중단된다.

---

## 에러 메시지

```
ValueError: Expected parameter probs (Tensor of shape (1, 3600)) of distribution
MaskableCategorical(probs: torch.Size([1, 3600]), logits: torch.Size([1, 3600]))
to satisfy the constraint Simplex(), but found invalid values:
tensor([[9.2541e-02, 1.5174e-07, 6.9758e-03,  ..., 1.0910e-04, 1.0910e-04,
         1.0910e-04]])
```

### 스택 트레이스

```
File "planning/packer.py", line 711, in pack
    fit_result = self.packingModel.stack(self.current_bin, self.items_list)
File "planning/RL/PalletFit_RL/rl_adapter.py", line 136, in stack
    action, _ = self.model.predict(...)
File "sb3_contrib/ppo_mask/ppo_mask.py", line 307, in predict
    return self.policy.predict(...)
File "sb3_contrib/common/maskable/policies.py", line 266, in _predict
    return self.get_distribution(...).get_actions(...)
File "sb3_contrib/common/maskable/policies.py", line 366, in get_distribution
    distribution.apply_masking(action_masks)
File "sb3_contrib/common/maskable/distributions.py", line 68, in apply_masking
    super().__init__(logits=logits)
File "torch/distributions/categorical.py", line 85, in __init__
    super().__init__(batch_shape, validate_args=validate_args)
File "torch/distributions/distribution.py", line 77, in __init__
    raise ValueError(...)
ValueError: Expected parameter probs ... to satisfy the constraint Simplex()
```

---

## 원인 분석

### 1. 에러 발생 위치: `MaskableCategorical.apply_masking`

`sb3-contrib`의 `MaskableCategorical`은 action masking을 지원하는 PyTorch Categorical 분포의 확장 클래스다.  
`apply_masking(masks)` 내부 로직:

```python
def apply_masking(self, masks):
    # 유효하지 않은 action의 logit을 -1e8으로 교체
    logits = th.where(masks, self._original_logits, HUGE_NEG)

    # 새 logits으로 Categorical 재초기화  ← 여기서 에러 발생
    super().__init__(logits=logits)

    # 캐시된 probs 강제 갱신
    self.probs = logits_to_probs(self.logits)
```

### 2. PyTorch 2.11에서의 변경 사항

| 버전 | `Distribution._validate_args` 기본값 |
|------|--------------------------------------|
| PyTorch < 2.11 | `False` |
| PyTorch >= 2.11 | **`True`** |

PyTorch 2.11부터 `torch.distributions.Distribution._validate_args = True`가 **기본값**으로 변경되었다.  
`validate_args=True` 상태에서는 분포 초기화 시 파라미터 유효성을 검증한다.

```
Categorical.arg_constraints = {
    'probs': constraints.simplex,   # ← 모든 값 >= 0, 합 = 1
    'logits': constraints.real_vector
}
```

### 3. `probs`가 `__dict__`에 캐시되는 메커니즘

`MaskableCategorical.__init__` 실행 순서:

```
1. super().__init__(logits=action_logits)       # Categorical 초기화
2. self._original_logits = self.logits          # 원본 logits 저장
3. self.apply_masking(None)                     # 내부 초기화용 호출
   ├─ super().__init__(logits=original_logits)  # probs는 아직 __dict__에 없음 → 검증 SKIP
   └─ self.probs = logits_to_probs(self.logits) # ← probs가 __dict__에 저장됨!
```

**이 시점에서 `self.probs`는 `__dict__`에 존재한다.**

이후 외부에서 `distribution.apply_masking(action_masks)` 호출:

```
4. masked_logits = th.where(masks, original_logits, -1e8)
5. super().__init__(logits=masked_logits)
   └─ Distribution.__init__() 검증 수행
      ├─ 'logits': __dict__에 있음 → real_vector 검증 → PASS
      └─ 'probs':  __dict__에 있음 (3번 단계에서 캐시됨) → Simplex 검증!
                   probs에 NaN이 있으면 → (NaN >= 0) == False → FAIL ❌
```

### 4. NaN의 발생 경로

첫 번째 스텝: bin이 비어있어 `items_topk`가 전부 0벡터 → 신경망 출력이 정상.

두 번째 스텝: bin에 1개 아이템이 배치된 상태 → `_encode_item_from_obj()`가 배치된 아이템의 실제 feature를 계산.

```
select_topk_ids(bin_obj)
  └─ _encode_item_from_obj(placed_item, bin_obj)
       ├─ get_direction_overlap()   # 방향별 gap/overlap 계산
       ├─ score_ez_distribution()   # 높이 분포 점수
       ├─ get_score_Guillotine()    # 기요틴 점수
       └─ estimate_support_ratio_safe()  # 지지율 계산
            └─ clip01(NaN) = NaN   # np.clip은 NaN을 전파함!
```

`numpy.clip`은 NaN을 그대로 통과시킨다:
```python
>>> np.clip(float('nan'), 0.0, 1.0)
nan
```

observation의 NaN → 신경망 출력 NaN → `logits_to_probs(NaN) = NaN` → `self.probs`에 NaN 캐시 → Simplex 검증 실패.

---

## 해결 방법

### Fix 1: `Distribution._validate_args = False` 설정 (호환성 fix)

`rl_adapter.py` 모듈 로드 시 PyTorch 2.11 이전 동작으로 복원:

```python
# planning/RL/PalletFit_RL/rl_adapter.py

from torch.distributions import Distribution

# PyTorch 2.11+에서 validate_args=True가 기본값으로 바뀌어
# MaskableCategorical에서 Simplex 검증 오류 발생 → False로 복원
Distribution._validate_args = False
```

- 학습 환경(PyTorch < 2.11)과 동일한 동작으로 맞춤
- `MaskableCategorical`의 distribution 검증 과정을 우회

### Fix 2: Observation NaN/Inf 방어 코드 (근본 fix)

`rl_adapter.py`의 `stack()` 메서드에서 `model.predict` 호출 직전:

```python
# NaN/Inf 방어: observation에 비정상 값이 있으면 0으로 교체
for k, v in obs.items():
    if isinstance(v, np.ndarray) and not np.isfinite(v).all():
        obs[k] = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
```

- bin에 아이템이 쌓이면서 feature 계산 중 NaN이 생성되는 경우를 포착
- 모델에 NaN이 입력되는 것 자체를 차단

### Fix 3: 관련 config 불일치 수정 (선행 작업)

이 에러에 앞서 config와 학습된 모델 간 shape 불일치도 수정이 필요했다:

| 설정값 | 잘못된 값 | 올바른 값 | 영향 |
|--------|-----------|-----------|------|
| `PREVIEW_MAX` | `5` | `10` | `preview_queue` shape `(5,4)` vs `(10,4)` |
| `ACTION_MAX_CANDIDATES` | `1200` | `360` | `act_mask` shape `(6000,)` vs `(3600,)` |

`config.py` 수정:
```python
PREVIEW_MAX = 10               # 모델 학습 시 사용한 값
ACTION_MAX_CANDIDATES: int = 360    # TOTAL = 10 * 360 = 3600
```

---

## 전체 에러 해결 순서

```
1. Git LFS로 weight 파일 다운로드
   ppo_ckpt_1265856_steps.zip: 134 bytes (LFS pointer) → 119 MB (실제 파일)
   → git lfs pull

2. config.py PREVIEW_MAX 수정: 5 → 10
   에러: "Unexpected observation shape (5, 4), please use (10, 4)"

3. config.py ACTION_MAX_CANDIDATES 수정: 1200 → 360
   에러: "Unexpected observation shape (12000,), please use (3600,)"

4. Distribution._validate_args = False + NaN 방어 코드 추가
   에러: "Expected parameter probs ... to satisfy the constraint Simplex()"
```

---

## 환경 정보

```
Python      : 3.10
PyTorch     : 2.11.0+cu130
sb3-contrib : 2.7.0
```

---

## 재발 방지

- 모델을 재학습하거나 weight를 교체할 때는 반드시 `config.py`의 `PREVIEW_MAX`, `ACTION_MAX_CANDIDATES`를 모델 학습 시 설정과 일치시킬 것
- PyTorch 버전을 업그레이드할 경우 `Distribution._validate_args` 기본값 변경 여부를 확인할 것
- 새 PyTorch 버전에서 학습 시에는 `Distribution._validate_args = False`를 학습 코드에도 명시할 것
