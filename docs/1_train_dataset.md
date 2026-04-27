# Train Dataset — 평가(Evaluation) 속도 분석 및 개선 방향

---

## 1. 문제 현상

학습 중 주기적으로 호출되는 `agent.evaluation()` 함수가,  
에이전트가 아이템을 잘 쌓을수록 (SU가 높을수록) 점점 오래 걸린다.  
3 에피소드 평가가 몇 분씩 소요되어 학습 전체가 느려지는 원인이 된다.

---

## 2. 평가 루프 구조

`evaluation()` 의 내부 루프는 다음 4단계를 매 step 반복한다.

```
while len(queue) > 0:
    ① rebuild_candidates(check=True)   ← 핵심 병목
    ② build_observation_deque()        ← 2번째 병목
    ③ model.predict()                  ← 빠름 (GPU inference)
    ④ place_by_action() + simplify()   ← 3번째 병목
```

아이템이 N개 쌓인 시점에서 각 단계의 비용:

| 단계 | 단계별 비용 | N이 커질수록 |
|---|---|---|
| ① `rebuild_candidates` | O(N × preview_cnt) | ↑↑ 지배적 |
| ② `build_observation_deque` | O(N log N) | ↑ 증가 |
| ③ `model.predict` | O(1) (상수) | 무관 |
| ④ `simplify` (per step) | O(N) | ↑ 증가 |
| ⑤ `render` (per episode) | 고정 5~30초 | 무관 |

에이전트가 50개를 잘 쌓으면, 50번째 step은 1번째 step보다 수십 배 느리다.  
전체 에피소드 비용 ≈ **O(total_items² × log(total_items))**

---

## 3. 원인 분석

### 3-1. [가장 큰 원인] EDP 후보 생성이 preview_cnt(10)번 중복 호출

`rebuild_candidates` 내부에서 preview 아이템 10개 각각에 대해 `get_action_mask`를 호출한다.  
`get_action_mask`는 내부적으로 `Edge_Projection(bin_obj)`를 매번 실행한다.

```python
# act_builder.py: rebuild_candidates 내부
for slot, item in look_pairs:          # 10번 반복
    if check:
        m_i, c_i = get_action_mask(bin, item, cands_option)
        # ↑ 내부에서 Edge_Projection(bin_obj) 호출 ← 매번 동일한 결과!
```

**`Edge_Projection(bin_obj)`는 bin 상태가 동일하면 10번 호출해도 항상 같은 pivot 목록을 반환한다.**  
같은 결과를 10번 계산하고 있다.

`Edge_Projection` 내부 비용:
- N개 아이템의 모든 edge에서 `project_lines_*_to_pivots` 호출 → **O(N) pivot 생성**
- `merge_close_pivots`: 생성된 pivot들 병합 → **O(P log P)**, P ≈ O(N)

그리고 생성된 P개 pivot 각각에 `checkPivot_R` (충돌 검사) 적용:
- checkPivot 1회 = R-tree 탐색 O(log N) + 충돌 검사
- P개 pivot × 10개 아이템 = **O(N × log N × 10)** / step

N=50일 때, step 50은 step 1보다 `50 × 10 = 500`배 이상의 비용이 든다.

---

### 3-2. 관측 빌드 (`build_observation_deque`) 비용 증가

```python
# obs_builder.py
ids = select_topk_ids(bin_obj, 64)          # get_visible_items_topdown: O(N log N)
for iid in ids:
    _encode_item_from_obj(it, bin_obj)       # get_direction_overlap: R-tree query × 64
```

- `get_visible_items_topdown`: N개 아이템을 정렬 + 임시 R-tree 구축 → **O(N log N)**
- `get_direction_overlap` × 64회: 방향별 gap/overlap 계산 → **O(64 × log N)**

N이 커질수록 매 step 비용이 증가.

---

### 3-3. `simplify()` 매 step 호출

```python
# agent.py: evaluation loop
if ok:
    bin_obj.simplify()    # ← 성공할 때마다 호출
```

`simplify`는 내부에서 최대 50번 반복하며 `post_merge`로 아이템 병합을 시도한다.  
병합이 없더라도 N개 아이템을 한 번씩 스캔 → **O(N) guaranteed per call**.

---

### 3-4. `render` 매 에피소드 호출 (렌더링 고정 비용)

```python
# agent.py: evaluation loop
bin_obj.render(save=True, save_path=str(render_dir), ...)
```

에피소드 완료 후 매번 렌더링 저장. Open3D/matplotlib 기반이므로 **5~30초 고정 비용**.  
3 에피소드 × 30초 = 최대 90초가 EDP/obs 비용과 무관하게 추가.

---

### 3-5. 에피소드 상한 없음

학습 초기에는 agent가 금방 실패해서 빠르지만, SU가 올라갈수록 에피소드가 길어진다.  
현재 `max_steps_per_episode=200`이 env에 설정되어 있으나, **evaluation 루프에는 이 상한이 없다.**

```python
while len(queue) > 0:    # ← 종료 조건이 오직 queue가 빌 때
    ...
```

---

## 4. 개선 방안

### 방안 A. EDP 중복 호출 제거 (가장 효과 큼, 약 10×↑)

`rebuild_candidates` 안에서 `Edge_Projection`을 한 번만 호출하고, 결과를 모든 preview 아이템에 재사용.

**현재 코드 구조 (문제):**
```python
for slot, item in look_pairs:          # preview_cnt(10)번
    m_i, c_i = get_action_mask(bin, item)  # 내부에서 EDP 매번 재실행
```

**개선 후 구조:**
```python
# EDP는 bin 상태 기준으로 한 번만
shared_candidates = Edge_Projection(bin_obj)   # 1번만

for slot, item in look_pairs:                  # preview_cnt번
    feasible = []
    for pv in shared_candidates:
        code, _ = checkPivot_R(bin_obj, item, [pv.x, pv.y, pv.z], pv.rt, apply_margin=True)
        if code == SUCCESS:
            feasible.append(pv)
    # feasible로 mask/cands 구성
```

`checkPivot_R`은 아이템별로 달라서 재사용 불가지만, EDP pivot 생성 자체는 bin 상태만 보므로 공유 가능.

---

### 방안 B. eval 루프에 step 상한 추가 (즉시 적용, 효과 큼)

```python
MAX_EVAL_STEPS = 150   # 에피소드당 최대 step 수

step_count = 0
while len(queue) > 0:
    step_count += 1
    if step_count > MAX_EVAL_STEPS:
        break
    ...
```

평가 목적은 SU 측정이므로, 150 step 안에 대부분의 아이템을 처리하면 충분.  
(실험 데이터셋 기준 아이템 수가 보통 30~80개이므로 150이면 여유 있음)

---

### 방안 C. eval 중 render 비활성화 (즉시 적용, 30~90초 절감)

학습 중 평가는 SU 수치만 필요. 렌더링은 최종 평가나 일정 주기에만 수행.

```python
# evaluation() 파라미터에 save_render 추가
def evaluation(self, episodes: int = 3, save_render: bool = False):
    ...
    if save_render:          # ← 기본 False
        bin_obj.render(save=True, ...)
```

또는 eval 호출 횟수로 제어:
```python
# N번에 한 번만 render
self._eval_render_count = getattr(self, "_eval_render_count", 0) + 1
should_render = (self._eval_render_count % 10 == 0)
```

---

### 방안 D. eval 중 `simplify` 비활성화

`simplify`는 학습 중 observation 품질을 위한 것.  
evaluation에서는 아이템을 빠르게 쌓고 SU만 측정하면 되므로 생략 가능.

```python
if ok:
    # bin_obj.simplify()    ← eval 중 생략
    pass
```

---

### 방안 E. eval 평가 에피소드 수 조정

현재 3개로 고정. 에피소드 수를 줄이면서도 다양한 데이터셋을 커버하려면:
- 학습 중 eval: **3~5개**, step cap 150
- 주기적 full eval (예: 10회에 1번): **10~20개**, render 포함

```python
# EvalAndAdaptiveCallback에서
metrics = self.eval_fn()           # fast eval (3 ep, no render)
if self._eval_count % 10 == 0:
    metrics_full = self.eval_fn(episodes=10, save_render=True)  # full eval
```

---

### 방안 F. 평가 루프를 `eval_env` 기반으로 교체 (구조 개선)

현재 `evaluation()`은 `PalletFitEnv`와 별개의 custom loop로 구현되어 있다.  
`self.eval_env`(DummyVecEnv 1개)를 사용하면 env 내부의 최적화를 그대로 활용할 수 있고,  
추후 병렬 eval env로 확장하기도 쉽다.

```python
# 현재: custom 루프
while len(queue) > 0:
    rebuild_candidates(...)
    build_observation_deque(...)
    model.predict(...)
    place_by_action(...)

# 개선: eval_env 사용
obs, _ = self.eval_env.reset()
done = [False]
while not all(done):
    action_masks = get_action_masks(self.eval_env)
    action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
    obs, reward, done, info = self.eval_env.step(action)
```

---

## 5. 권장 적용 순서

| 순서 | 방안 | 예상 효과 | 난이도 | 상태 |
|---|---|---|---|---|
| 1 | C. render 비활성화 | 30~90초 절감 (즉시) | 매우 쉬움 | ✅ 적용 |
| 2 | B. step 상한 추가 | 후반 step 누적 비용 차단 | 매우 쉬움 | ✅ 적용 |
| 3 | D. simplify 비활성화 | 각 step O(N) 절감 | 매우 쉬움 | ⏸ 보류 |
| 4 | A. EDP 중복 제거 | 약 10× 속도 개선 | 중간 (act_builder 수정) | ✅ 적용 |
| 5 | E. 에피소드 수 분리 | 통계 안정성 + 속도 균형 | 쉬움 | ⏸ 보류 |
| 6 | F. eval_env 기반 교체 | 코드 통일 + 병렬화 기반 | 중간 | ⏸ 보류 |

---

## 6. 적용된 변경사항 (2026-04-23)

### 6-1. EDP 중복 제거 — `act_builder.py`

`get_action_mask()`에 `precomputed_candidates` 파라미터 추가 (기본 None, 기존 호출부 영향 없음).
`rebuild_candidates()`에서 `check=True`일 때 `Edge_Projection(bin)`을 **preview_cnt번이 아닌 1번만** 호출하고 모든 preview 아이템에 재사용.

```python
# act_builder.py: rebuild_candidates
shared_candidates = None
if check and look_pairs:
    if cands_option == 'CP':
        shared_candidates = get_pivots_cp(bin)
    elif cands_option == 'EP':
        shared_candidates = get_pivots_ep(bin)
    elif cands_option == 'EMS':
        shared_candidates = get_pivots_ems(bin)
    else:  # 'EDP'
        shared_candidates = Edge_Projection(bin)

for slot, item in look_pairs:
    if check:
        m_i, c_i = get_action_mask(bin, item, cands_option,
                                   precomputed_candidates=shared_candidates)
```

> 기존에는 `get_action_mask`가 내부적으로 `Edge_Projection`을 호출했기 때문에 preview 10개에 대해 총 10번 재계산했음. 이제 1번만 계산 → **EDP 생성 비용 약 10× 감소**.

### 6-2. step 상한 + render 제어 — `agent.py`

`evaluation()`에 파라미터 추가:

```python
def evaluation(
    self,
    episodes: int = 3,
    max_eval_steps: int = 150,   # 에피소드당 최대 step
    render_episodes: int = 1,    # 렌더링할 에피소드 수 (기본 1 = 첫 번째만)
) -> Dict[str, float]:
```

루프 내부:
```python
step_count = 0
while len(queue) > 0:
    if step_count >= max_eval_steps:
        break
    step_count += 1
    ...
```

에피소드 종료 후:
```python
# 모든 bin 상태를 캐시 (수동 재렌더용)
self._last_eval_bins.append(bin_obj)

# 처음 N개만 자동 렌더
if i < render_episodes:
    bin_obj.render(save=True, ...)
```

### 6-3. 사후 수동 렌더 기능 — `agent.py`

평가 중 첫 에피소드만 자동 렌더하므로, 나머지를 나중에 시각적으로 확인할 수 있는 헬퍼 추가:

```python
def render_last_eval(self, indices=None, save_dir: Optional[str] = None) -> None:
    """직전 evaluation의 bin들을 수동 렌더."""
```

사용법:
```python
agent.evaluation()                         # ep000만 자동 렌더
agent.render_last_eval()                   # 나머지 전부 렌더
agent.render_last_eval(indices=[1, 2])     # 특정 인덱스만
agent.render_last_eval(save_dir="/tmp/x")  # 다른 폴더에 저장
```

---

## 7. 남은 과제

현재 적용되지 않은 항목들은 필요 시 추가 적용 가능:

- **D. `simplify()` 비활성화**: 각 step의 O(N) merge 비용. eval 중에는 SU 측정에 영향 없으므로 생략 고려.
- **E. 에피소드 수 분리**: 빠른 eval(3~5ep) + 주기적 full eval(10~20ep) 이원화.
- **F. `eval_env` 기반 통합**: 현재 custom 루프와 `PalletFitEnv` 로직이 분리되어 있음. 통합 시 중복 제거 + 병렬 평가 확장 가능.

---

## 8. 요약

```
느린 이유:
  - 쌓인 아이템 N개 → EDP가 O(N) pivot 생성 → checkPivot O(N log N)
  - 이걸 preview_cnt(10)번 반복 → 10× 중복        ← ✅ fix
  - 매 step마다 simplify() O(N)                    ← ⏸ 보류
  - 에피소드 종료 후 render 5~30초                 ← ✅ fix (ep0만)
  - step 상한 없음 → 잘 쌓을수록 에피소드가 끝없이 → ✅ fix (150)

적용된 fix:
  A. EDP를 한 번만 호출하고 preview 아이템들이 공유
  B. eval 루프에 max_eval_steps=150 추가
  C. render_episodes=1로 첫 에피소드만 렌더, render_last_eval()로 사후 렌더 지원
```
