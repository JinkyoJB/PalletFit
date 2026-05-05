# Train Dataset — 평가(Evaluation) 속도 분석 및 개선 방향

> **2026-05-04 업데이트**: 이 문서가 처음 작성될 때(2026-04-23)의 *custom serial 평가 루프*는 **완전히 폐기**됐고, 평가가 `SubprocVecEnv` 기반으로 재작성됐습니다. Sec 2~5의 분석은 **당시 컨텍스트의 역사 기록**이고, 현재 상태는 **Sec 9 (전면 개편 이후)** 와 `2_env.md` Sec 6를 우선 참조하세요.

---

## 1. 문제 현상 (2026-04-23 시점)

학습 중 주기적으로 호출되는 `agent.evaluation()` 함수가,  
에이전트가 아이템을 잘 쌓을수록 (SU가 높을수록) 점점 오래 걸린다.  
3 에피소드 평가가 몇 분씩 소요되어 학습 전체가 느려지는 원인이 된다.

---

## 2. 평가 루프 구조 (2026-04-23 시점, 이후 폐기)

> ⚠️ 이 절의 custom 루프는 **2026-05-01 옵션 D 리팩토링 시 완전 제거**됨. Sec 9 참조.

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

## 5. 권장 적용 순서 (상태 갱신: 2026-05-04)

| 순서 | 방안 | 예상 효과 | 난이도 | 상태 |
|---|---|---|---|---|
| 1 | C. render 비활성화 / 제어 | 30~90초 절감 (즉시) | 매우 쉬움 | ✅ 적용 (2026-04-23, Sec 6) → 그 후 GIF로 재설계 (2026-04-27, Sec 9) |
| 2 | B. step 상한 추가 | 후반 step 누적 비용 차단 | 매우 쉬움 | ✅ 적용 (2026-04-23) → env의 `max_steps_per_episode`로 흡수 (Sec 9) |
| 3 | D. simplify 비활성화 | 각 step O(N) 절감 | 매우 쉬움 | ✅ 무관 — custom 루프 폐기로 simplify 호출 자체 사라짐 (Sec 9) |
| 4 | A. EDP 중복 제거 | 약 10× 속도 개선 | 중간 (act_builder 수정) | ✅ 적용 (2026-04-23) → env step에서 그대로 활용됨 |
| 5 | E. 에피소드 수 분리 | 통계 안정성 + 속도 균형 | 쉬움 | ⏸ 보류 — 현재 `n_envs_eval=5` 1 라운드로 충분 판단 |
| 6 | F. eval_env 기반 교체 | 코드 통일 + 병렬화 기반 | 중간 | ✅ 적용 (2026-04-27, Sec 9) — SubprocVecEnv N envs로 병렬 평가 |

---

## 6. 적용된 변경사항 — 1차 (2026-04-23, custom 루프 시점)

> ⚠️ 이 절의 fix들은 **custom evaluation 루프**를 전제로 한 것. 2026-04-27의 VecEnv 전면 개편(Sec 9)에서 custom 루프 자체가 사라지면서 6-2/6-3은 코드 형태가 바뀌었지만 **의도(step cap, 첫 ep 시각화)는 새 구조에 그대로 흡수**됨. 6-1(EDP 중복 제거)는 `act_builder.py` 변경이라 그대로 유효.

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

## 7. 남은 과제 (2026-05-04 갱신)

원래 남은 과제 3개 중 2개가 Sec 9의 VecEnv 개편에 흡수되거나 무관해졌고, 한 항목만 보류:

- ✅ **D. `simplify()` 비활성화**: custom 루프 폐기로 `simplify` 호출 자체가 사라짐 → **자동 무관**.
- ✅ **F. `eval_env` 기반 통합**: 2026-04-27 VecEnv 전면 개편으로 적용됨 (Sec 9, `2_env.md` Sec 6).
- ⏸ **E. 에피소드 수 분리**: 현재 `n_envs_eval=5` 한 라운드 = 5 episode가 학습 중 평가로 충분. full eval(10~20ep) 이원화는 **학습 결과의 SU 분산이 불안정할 때** 검토.

---

## 8. 요약 (2026-05-04 갱신)

```
=== 1차 fix (2026-04-23, custom 루프 시점) ===
  A. EDP 한 번만 호출, preview 아이템들이 공유          → 여전히 유효 (act_builder.py)
  B. eval 루프에 max_eval_steps=150                      → custom 루프 폐기 후 env의 max_steps_per_episode로 흡수
  C. render_episodes=1 + render_last_eval() 사후 렌더    → GIF 자동 저장 방식으로 재설계

=== 2차 전면 개편 (2026-04-27 ~ 05-01, Sec 9) ===
  - custom serial 루프 → SubprocVecEnv 라운드-로빈 평가 (n_envs_eval = 5)
  - 0번 워커만 is_render_env=True → eval_renders/ep{idx:04d}_SU{su:.3f}.gif 자동 저장
  - simplify/render/step-cap 관련 옵션이 모두 env 안으로 흡수 → agent.py의 evaluation()은 ~70줄
  - _ep_reward_terms를 per-step delta 누적으로 정합 (sum == ep_return) → TB 메트릭 의미 정확
```

→ 결과적으로 평가는 wall time이 ~`n_envs_eval`× 단축되었고, 코드는 env step 로직 한 군데로 통일됨.

---

## 9. 전면 개편 — Custom 루프 폐기 + VecEnv 평가 (2026-04-27 ~ 05-01)

### 9-1. 개편 동기

Sec 6의 1차 fix가 잘 작동하긴 했지만 **근본 문제 2가지가 남아 있었음**:

1. **`evaluation()`이 `PalletFitEnv`와 별개의 custom 루프**라서 env 안의 최적화(EDP dedup, getDimension 캐싱, face cache 등)를 자동으로 받지 못함. 매번 `Packer`/items를 직접 만들고 builder 함수를 직접 호출하는 ~200줄 직렬 루프.
2. **단일 프로세스 직렬 평가**라 `eval_env`(DummyVecEnv 1개)가 실질적으로 사용되지 않고 메인 프로세스에서 한 에피소드씩 처리. n_envs_eval=5로 늘려도 효과 0.

→ Sec 5의 F번(eval_env 기반 통합) + 추가로 SubprocVecEnv 병렬 평가 + 0번 워커 GIF 캡처를 한 번에 묶어 적용.

### 9-2. 새 구조 (현재 상태)

**`agent.make_eval_env`**:
```python
def make_eval_env(*, n_envs, base_seed, tb_log_dir):
    thunks = [
        make_single_env(base_seed + i, tb_log_dir, is_render_env=(i == 0))
        for i in range(n_envs)
    ]
    venv = SubprocVecEnv(thunks, start_method="spawn")
    return VecMonitor(venv, filename=str(Path(tb_log_dir) / "monitor_eval"))
```
- `backend` 인자 제거 → SubprocVecEnv 고정. spawn 방식.
- 0번 thunk만 `is_render_env=True` → 그 워커가 매 평가 에피소드를 GIF로 저장.

**`PalletFitEnv` (env.py)**:
- `_capture_frame()` / `_flush_gif()`: 매 successful placement에 frame 누적, 종료(`terminated or truncated`) 시 GIF 저장.
- 캡처 활성화는 **`is_render_env` AND `_pending_plan` 소비 AND tag.startswith("eval")** 일 때만 → SB3 자동 재리셋(같은 plan 반복) 시 중복 GIF 방지.
- 이전 `save_render` PNG 헬퍼 + `_save_render_on_done` 분기는 dead code로 삭제(plan_maker가 발급하던 `eval_offline`/`eval_online`/`eval_tsg` tag와 매칭 안 되던 사실상 죽은 코드였음).

**`agent.evaluation()`** (이전 ~200줄 → ~70줄):
```python
@th.no_grad()
def evaluation(self, episodes=None, max_eval_steps=200) -> Dict[str, float]:
    self.model.policy.set_training_mode(False)
    n_envs = int(self.eval_env.num_envs)
    if episodes is None:
        episodes = n_envs

    plans = self._cached_eval_plans  # 재현성용 캐시
    noop_idx = int(self.eval_env.get_attr("NOOP_IDX", indices=[0])[0])

    su_list, packed_list = [], []
    plan_idx = 0
    while plan_idx < episodes:
        round_size = min(n_envs, episodes - plan_idx)
        # 라운드 plan을 워커들에 주입
        for i in range(round_size):
            payload = env_plan_to_payload(plans[plan_idx + i])
            self.eval_env.env_method("apply_plan", payload, indices=[i])

        obs = self.eval_env.reset()
        done_mask = np.zeros(n_envs, dtype=bool)
        done_mask[round_size:] = True   # 미사용 워커는 done 취급

        for _ in range(max_eval_steps):
            if done_mask.all(): break
            masks = get_action_masks(self.eval_env)
            actions, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
            actions = np.asarray(actions, dtype=np.int64)
            if done_mask.any():
                actions[done_mask] = noop_idx   # 끝난 워커는 NOOP 패딩
            obs, _, dones, infos = self.eval_env.step(actions)
            for i in range(round_size):
                if dones[i] and not done_mask[i]:
                    done_mask[i] = True
                    info = infos[i] if isinstance(infos[i], dict) else {}
                    su_list.append(float(info.get("SU", 0.0)))
                    packed_list.append(int(info.get("packed_count", 0)))
        plan_idx += round_size

    self.model.policy.set_training_mode(True)
    return {"SU": float(np.mean(su_list)), "packed": float(np.mean(packed_list))}
```

핵심 패턴:
- **plan 주입 → reset → batched predict → step**: env step 로직 한 군데로 신뢰.
- **NOOP 패딩**: 라운드에서 일찍 끝난 워커가 자동 재리셋되어도 `_pending_plan` 비어 있어 GIF 중복 안 생김.
- **마스크 조회**: `sb3_contrib.common.maskable.utils.get_action_masks(self.eval_env)`로 워커별 action_masks 받아 batched 추론.

### 9-3. 1차 fix들이 새 구조에 흡수된 방식

| 1차 fix (2026-04-23) | 새 구조에서 어떻게 보존됐나 |
|---|---|
| **A. EDP dedup** | 그대로 유효. `act_builder.rebuild_candidates`의 변경이라 env step에서 자동으로 활용됨. |
| **B. step cap (150)** | env의 `max_steps_per_episode`(plan에서 설정)로 이전. evaluation의 `max_eval_steps=200`은 라운드 안전장치(워커가 안 끝나는 무한 루프 방지). |
| **C. render 1 ep만** | 0번 워커만 `is_render_env=True` → 평가 시 자동으로 N개 라운드의 모든 ep 중 0번 워커 분만 GIF 저장. `render_last_eval` 헬퍼는 메인 프로세스가 bin 객체 안 들고 있으니 삭제. |
| **D. simplify 비활성화** | custom 루프가 사라져서 `bin_obj.simplify()` 호출 자체가 없음 → 자동 무관. |

### 9-4. 추가 부수 개선 (이 시기에 같이 들어간 것들)

새 구조와 함께 들어간 인접 변경. 자세한 내용은 `2_env.md` Sec 5/6/7 참조:

- **Item.getDimension / face_cache 캐싱** (2_env.md Sec 7-6): `_calculate_support_ratio_geometric`이 ~50-100× 빨라지고, 그 cascade로 `get_direction_overlap`, `_get_contact_score` 등 모든 hot path가 함께 가속. env step 평균 ~155ms → ~115ms.
- **종료 경로 rebuild/obs 스킵** (2_env.md Sec 5-4): `_finalize_step`에서 `terminated=True`일 때 `_safe_rebuild_candidates` + `build_obs` 호출 모두 생략. SB3가 다음 obs 안 쓰니까 안전.
- **Invalid reset 스킵** (2_env.md Sec 5-4): `_build_items_from_plan` 실패 시 `_make_zero_obs()` + NOOP-only mask로 즉시 리턴.
- **`_ep_reward_terms` per-step delta 누적** (3_rewards.md Sec 7-2): TB의 `train/reward_ep/*`가 진짜 reward 기여도가 됨 → 평가 외 학습 모니터링 정확도 ↑.

### 9-5. 측정 결과

| 평가 에피소드 수 | 이전 (Custom 직렬, 1 env) | 현재 (Subproc N envs, N=5) | 속도 비 |
|---|---|---|---|
| `n_envs_eval` (=5) | t × 5 | max ≈ t | ~5× |
| `2 × n_envs_eval` | t × 10 | t × 2 | ~5× |

> Subproc 시작 비용은 학습 시작 1회로 amortize됨(eval_env는 재사용). 라운드 종료 대기 중 NOOP 패딩 step은 워커당 최대 1개 step 정도라 무시 수준.

### 9-6. 남은 한 가지 검토 (E. 에피소드 수 분리)

현재 `n_envs_eval=5` 한 라운드 = 5 episode. 통계적으로 충분한지 **학습 결과의 SU 분산을 보고 판단**:
- 동일 모델로 여러 라운드 측정 시 SU std가 0.05 이상이면 → episodes를 10~15로 늘려 2~3 라운드 평가 검토.
- 현재 라운드 분배 알고리즘이 episodes > n_envs일 때 자동으로 multi-round 처리하므로 코드 변경 없이 호출 인자만 바꾸면 됨.

### 9-7. Edge_Projection 내부 최적화 (2026-05-04)

**배경**:
- Sec 6-1에서 "EDP를 1번만 호출(call-site dedup)" 적용 후에도 `Edge_Projection` 자체가 **bin items=15 기준 ~33ms/call**로 여전히 무거움.
- 알고리즘(Liang-Barsky 기반 boundary_projection)은 빠른데(1.2s/100 calls = 0.0025ms/call) **호출 패턴**이 비효율:
  - 5개 sub 함수가 각자 `bin.get_visible_items_topdown()`, `bin.get_all_items()`, `[*..., bin]` 빌드 → 동일 작업 5× 중복.
  - 각 sub 함수가 회전 4종 × 후보점마다 `Pivot` 객체를 만든 뒤 끝에서 dedup해 95% 폐기 → 객체 생성 낭비.
  - 좌표 정리/dedup key 만들기에 `round(x, 3)`이 호출당 ~6,600회 (총 65만회).

**진단 (cProfile, bin items=15, 100회 호출)**:

| 항목 | 호출 수 | cumulative | 분석 |
|---|---|---|---|
| `Edge_Projection` (전체) | 100 | 5.10s = **33.5ms/call** | — |
| `Pivot.__init__` | 205,800 | 0.67s (13%) | 95% 중복 후 폐기 |
| `built-in round` | 659,500 | 0.65s (13%) | dedup key 생성 + 좌표 정리 |
| `get_visible_items_topdown` | 300 | 0.14s (3%) | 5개 sub 중복 호출 |
| `merge_close_pivots` | 100 | 0.56s (11%) | (사용자 결정으로 변경 안 함) |

**수정 (Phase 1.1 + 1.2)**:

1. **공통 데이터 사전 계산** (`act_builder.py`):
   ```python
   def Edge_Projection(bin):
       ...
       # 5 sub 모두 같은 데이터 → 1번만 계산해서 인자로 전달
       visible_items = bin.get_visible_items_topdown()
       plane_items   = [*bin.get_all_items(), bin]
       
       left_pivots  = project_lines_left_to_pivots(bin, visible_items, plane_items)
       front_pivots = project_lines_front_to_pivots(bin, visible_items, plane_items)
       down_pivots  = project_lines_down_to_pivots(bin, visible_items, plane_items)
       # 2left/2front는 down_pivots만 받으므로 인자 변경 없음
       ...
   ```

2. **Sub 함수 시그니처 확장** (`utils/pivot_generation.py`):
   ```python
   def project_lines_left_to_pivots(bin, visible_items=None, plane_items=None):
       if visible_items is None: visible_items = bin.get_visible_items_topdown()
       if plane_items is None: plane_items = [*bin.get_all_items(), bin]
       ...
   ```
   기본값 None → 외부 직접 호출 시 backward compat 유지.

3. **Pivot 생성 dedup BEFORE 객체 생성** (5개 sub 모두):
   ```python
   # 이전
   for rt in RotationType.BasicRotation[0]:
       pivots.append(Pivot(round(px,3), round(py,3), round(pz,3), rt, ...))
   # 끝에서 dedup → 95% 객체 폐기

   # 현재
   rpx, rpy, rpz = round(px, 3), round(py, 3), round(pz, 3)   # 한 번만 round
   for rt in RotationType.BasicRotation[0]:
       rt_key = tuple(map(float, rt))
       key = (rpx, rpy, rpz, rt_key)
       if key in seen: continue                                # dedup 먼저
       seen.add(key)
       pivots.append(Pivot(rpx, rpy, rpz, rt, ...))            # unique만 객체화
   ```
   `merge_close_pivots`은 그대로 — **사용자 컨셉**: margin 때문에 비슷해 보이는 pivot이 생겨도 엄연히 다른 값이라 합치지 않음.

**의도적으로 안 한 것**:
- `merge_close_pivots` 공간 인덱스화 (사용자 결정으로 보존).
- `boundary_projection` pre-filter 강화 (Phase 2 후보로 보류).
- 5 sub 함수 통합 (Phase 3, 위험 대비 이득 작음).

**측정 결과 (동일 시나리오, bin items=15)**:

| 항목 | 이전 | 현재 | 변화 |
|---|---|---|---|
| Edge_Projection wall time | 33.48 ms/call | **24.15 ms/call** | **−27.9%** |
| 전체 함수 호출 수 | 6.61M | 5.70M | −13.8% |
| `Pivot.__init__` 호출 | 205,800 | (top 15에서 사라짐) | ~95% ↓ |
| `round` 호출 | 659,500 | 481,900 | −27% |
| `get_visible_items_topdown` 호출 | 300 | 100 | 3× 감소 |
| 결과 pivot 수 (정확성) | 108 | **108** | 동일 ✓ |
| unique (x,y,z,rt) key 수 | 108 | 108 | 동일 ✓ |

**남은 hot spot (Phase 2 후보, 학습 결과 본 뒤 결정)**:
- `boundary_projection` 71,000 calls (~1.2s, 30%) — line × plane 조합 quadratic. plane_items 사전 sort + binary search로 후보군 좁히면 추가 감축 가능.
- `search_xyz` 12,800 calls (~0.8s, 20%) — R-tree intersection의 `set & set & set` 변환 비용. 더 작은 set부터 교집합하면 미세 절감.
- `_candidate_*_faces_line` 안의 plane 반복 — line별로 fresh iteration. plane으로 grouping하면 캐시 적중 ↑.

**효과**:
- env step 평균 ~24 ms 정도 단축 효과 (build_obs와 함께 Edge_Projection이 step의 큰 비중).
- 24 envs × 30 step × ~9ms 절감 = 학습 rollout당 ~6.5초 단축 추정.
- 코드 변경 위험: 매우 낮음 (결과 정확성 검증됨, 외부 호출 시그니처는 backward compat).
