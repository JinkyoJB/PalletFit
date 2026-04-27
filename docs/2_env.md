# Env 최적화 — PalletFitEnv

> 대상 파일: `planning/RL/PalletFit_RL/env.py`, `planning/RL/PalletFit_RL/reward_builder.py`
> 목적: 학습/평가 공통으로 쓰이는 환경의 step 비용을 낮춰 전체 학습 throughput을 끌어올림.

---

## 1. 배경

이전 `1_train_dataset.md`에서 evaluation 속도 문제를 분석하면서,
근본 원인이 **env 내부가 step마다 많은 비용을 지불하는 구조**임이 드러났다.

- eval loop (`agent.evaluation`)은 `PalletFitEnv`와 별개의 custom 루프지만,
  내부적으로 동일한 builder 함수(`rebuild_candidates`, `build_obs`, `place_by_action`)를 사용한다.
- 즉, **env의 step이 느리면 학습 rollout도 느리고 eval도 느리다.**
- EDP 중복 제거(Task 1)로 후보 생성 부분은 개선했지만, env의 step 경로 자체는 추가 최적화 여지가 있었다.

---

## 2. 현재 구현 구조 (2026-04-27 업데이트)

### 2-1. 핵심 상태

```python
self._N_max   = PREVIEW_MAX (10)
self._K       = ACTION_MAX_CANDIDATES (360)
self._TOTAL   = 3600
self.NOOP_IDX = 3599
self._prev_score: float = 0.0   # ★ 신규: reward delta 계산용 baseline

observation_space = Dict({
    items_topk:    (64, 22),
    items_mask:    (64,),
    globals:       (6,),
    preview_queue: (10, 4),
    act_mask:      (3600,),
    act_cands:     (3600, 4),
})
action_space = Discrete(3600)
```

### 2-2. reset() 흐름

1. plan 결정 (`_pending_plan`/`_plan_in_use`/기본)
2. `_build_binPacker_from_plan()` — bin/margin/preview_cnt 설정
3. `_build_items_from_plan()` — offline/online_type/tsg 모드로 아이템 로드
4. `_safe_rebuild_candidates(check=True)`
5. `build_obs(...)`
6. **★ `self._prev_score = build_reward(self._bin, placed_item=None)[0]`** — delta baseline 초기화

### 2-3. step() 흐름 (★ deepcopy 제거됨)

```
step(action):
  ① self._steps_in_ep += 1                       # deepcopy 없음
  ② if max_steps 초과: finalize(truncated)
  ③ if queue 비어있음: finalize(terminated, no_items)
  ④ if action == NOOP: finalize(terminated, NOOP)

  ⑤ place_by_action(...) 시도
     ├─ ok: queue 업데이트, finalize(placed_item=item)
     └─ fail:
         · _last_mask[action] = False
         · retry_count++
         · if all_masked or retry_count ≥ 100: finalize(RETRY_LIMIT)
         · else: return last_obs, -0.01, False, False, info  # 저비용 retry

  _finalize_step:
    if 실패 종료:
        reward, terms = get_failure_penalty(failure_code)        # ★ 분리
    else:
        curr_score, terms = build_reward(self._bin, placed_item) # ★ 단일 bin
        reward = curr_score - self._prev_score
        self._prev_score = curr_score - placement_only           # baseline 갱신
    
    _safe_rebuild_candidates(check=True)
    build_obs(...)
```

### 2-4. 병렬화 구성 (2026-04-27 업데이트)

- **학습**: `SubprocVecEnv` 24개 워커 (spawn 방식)
- **평가**: `SubprocVecEnv` `n_envs_eval`개 워커 (spawn 방식). 0번 워커만 `is_render_env=True`로 매 에피소드 GIF 저장. ✅

---

## 3. SubprocVecEnv vs DummyVecEnv

Stable-Baselines3의 환경 병렬화 두 가지 백엔드. `agent.py`에서 둘 다 사용 중.

### 3-1. DummyVecEnv

> "Vec 인터페이스만 갖춘 단일 프로세스 환경 묶음."

```python
from stable_baselines3.common.vec_env import DummyVecEnv

venv = DummyVecEnv([thunk1, thunk2, thunk3])
# 내부적으로 list[env]를 들고 있고, step()은 for env in envs: env.step()으로 순차 실행
```

| 특징 | 설명 |
|---|---|
| **동작 방식** | 메인 프로세스의 같은 Python 인터프리터에서 env들을 **순차 실행** |
| **데이터 전달** | 메모리 직접 공유 (직렬화 없음) → 매우 빠른 IPC |
| **GIL 영향** | Python GIL 때문에 진짜 병렬 실행 X |
| **장점** | 디버깅 쉬움 (스택트레이스, breakpoint 잘 통함), import 비용 1회만 |
| **단점** | env가 CPU-heavy면 throughput이 곧 1 env 속도와 같음 |
| **언제 쓰나** | env가 가볍고 GPU 추론이 병목일 때, 디버깅 시, n_envs=1~2일 때 |

**현재 PalletFit 평가에서**: `n_envs=1`이라 어차피 "병렬화"라 부를 게 없고 그냥 단일 env 래퍼.

### 3-2. SubprocVecEnv

> "각 env를 별도 프로세스에 띄우고 pipe로 통신."

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

venv = SubprocVecEnv([thunk1, thunk2, ..., thunk24], start_method="spawn")
# 24개의 자식 프로세스가 각자 PalletFitEnv 인스턴스를 보유
# step() 호출 시 모든 자식에 명령 broadcast → 응답 대기
```

| 특징 | 설명 |
|---|---|
| **동작 방식** | env마다 별도 프로세스 → **진짜 병렬 실행** (GIL 우회) |
| **데이터 전달** | obs/action을 **pickle로 직렬화 + pipe 전송** → IPC 비용 있음 |
| **시작 방식** | `fork` (Linux 기본, 메모리 공유 후 분기) / `spawn` (안전, 새 인터프리터) |
| **장점** | env가 CPU-heavy일수록 효과 큼 (`build_obs`, `rebuild_candidates`가 무거운 PalletFit에 적합) |
| **단점** | 자식 프로세스 시작 비용 (수 초), 직렬화 비용, 디버깅 어려움 |
| **언제 쓰나** | env가 무거울 때, n_envs ≥ 4 권장 (그 미만이면 IPC 오버헤드가 이득보다 큼) |

**현재 PalletFit 학습에서**: 24 워커 × spawn → CPU 24코어를 거의 다 쓰며 학습 rollout을 24× 가속.

### 3-3. 왜 평가에 DummyVecEnv를 썼었나? (히스토리)

이전엔 다음 이유로 Dummy 1개를 썼었다:
1. **디버깅 편의**: 평가 중 에러는 metric 신뢰성에 직결되므로 스택트레이스가 깔끔한 환경에서 돌리고 싶었음.
2. **학습이 정말 되는지 확인용**: 매 평가마다 첫 에피소드를 시각화(PNG render)하고 싶었음 → 메인 프로세스에서 bin 객체를 직접 들고 있어야 했음.
3. **에피소드 수가 적음** (3개): SubprocVecEnv 시작 비용이 평가 절대 시간 대비 컸음.

→ **2026-04-27 해결**: GIF 저장을 **0번 워커 내부**에서 처리(`bin.render(return_array=True)` 프레임 누적 + 에피소드 종료 시 imageio.mimsave)하면서, 메인 프로세스가 bin 객체를 들 필요가 없어짐. 평가도 SubprocVecEnv로 전환(아래 Sec 6 참고).

---

## 4. 핵심 병목 / 문제점 (현재 상태)

| # | 항목 | 영향 | 상태 |
|---|---|---|---|
| 3-1 | `copy.deepcopy(self._bin)` 매 step | ⭐⭐⭐ | ✅ **해결** (state-based reward로 교체) |
| 3-2 | `_safe_rebuild_candidates(check=True)` 종료 경로 호출 | ⭐⭐ | ✅ **해결** (terminated 시 rebuild/obs 스킵) |
| 3-3 | `build_obs` O(N log N) 매 step | ⭐⭐ | ✅ **해결** (Item.getDimension 캐싱 → ~18× 단축, Sec 5-5) |
| 3-4 | reward에 `cand_item=None` (stability/contact 무력화) | ⭐⭐⭐ | ✅ **해결** (`placed_item` 전달) |
| 3-5 | `max_retry_per_step=100` 과다 | ⭐ | ✅ **해결** (기본값 20으로 축소) |
| 3-6 | step 진입 직후 deepcopy → 조기 종료에도 부담 | ⭐ | ✅ **해결** (deepcopy 자체 제거) |
| 3-7 | 평가 환경이 DummyVecEnv 1개 → 순차 실행 | ⭐⭐ | ✅ **해결** (Sec. 6 — SubprocVecEnv + 0번 워커 GIF) |
| 3-8 | reset 경로 invalid items에서 rebuild 호출 불필요 | ⭐ | ✅ **해결** (NOOP-only + 0-채움 obs) |

> **3-1 / 3-4 / 3-6은 동일 커밋으로 해결됨**: `build_reward`를 single-bin state score 함수로 리팩토링하고, env가 step n과 n+1의 score 차이로 reward를 계산하는 구조로 변경. `placed_item`도 자연스럽게 전달 경로에 끼움.

---

## 5. 적용된 변경사항 (2026-04-27)

### 5-1. `reward_builder.py` — single-bin state score

```python
def build_reward(bin_obj, placed_item=None) -> Tuple[float, Dict[str, float]]:
    """현재 bin 상태의 절대 점수.
    env가 step n과 n+1의 차이로 reward를 계산."""
    # State-based (bin 상태에만 의존)
    r_su   = SU_WEIGHT  * bin_obj.SU * 10.0
    r_dead = -DEAD_WEIGHT * _dead_ratio(bin_obj) * 10.0
    r_bal  = BALANCE_WEIGHT * balance_term_capped(1.0, bin_obj, cand_item=None)

    # Placement-specific (placed_item이 있을 때만)
    r_alive = r_stab = r_contact = 0.0
    if placed_item is not None:
        r_alive   = ALIVE_BONUS
        r_stab    = (_calculate_support_ratio_geometric(bin_obj, placed_item) - 1.0) * STABILITY_PENALTY_WEIGHT
        r_contact = CONTACT_WEIGHT * (_get_contact_score(bin_obj, placed_item) / 10000.0)

    total = r_su + r_dead + r_bal + r_alive + r_stab + r_contact
    return total, {"eff_su":..., "eff_dead":..., "bal":..., "alive":..., "stab_soft":..., "qual_contact":...}


def get_failure_penalty(failure_code=None) -> Tuple[float, Dict[str, float]]:
    """실패 종료 시 페널티만 반환 (env가 직접 호출)."""
```

- **`r_su`, `r_dead`**: state-based지만 delta(`curr - prev`)는 기존 `Δ × weight`와 수학적으로 동일 (절대 항이 상쇄).
- **`r_bal`**: cap을 고정 reference(1.0)로 사용 → 매 step 영향이 일정 범위로 안정화.
- **`r_stability`, `r_contact`**: `placed_item` 전달로 **드디어 활성화**.

### 5-2. `env.py` — deepcopy 제거 + delta 흐름

```python
# __init__
self._prev_score: float = 0.0

# reset() 끝부분
initial_score, _ = build_reward(self._bin, placed_item=None)
self._prev_score = float(initial_score)

# step() 진입 — deepcopy 없음
self._steps_in_ep += 1
if 조기종료조건:
    return self._finalize_step(...)

# 성공 시
return self._finalize_step(..., placed_item=placed_item)

# _finalize_step
if terminated and not finished:
    reward, terms = get_failure_penalty(failure_code)
else:
    curr_score, terms = build_reward(self._bin, placed_item=placed_item)
    reward = curr_score - self._prev_score
    placement_only = terms["alive"] + terms["stab_soft"] + terms["qual_contact"]
    self._prev_score = curr_score - placement_only   # 다음 step의 baseline (state-only)
```

### 5-3. 부수 효과 — `no_items` terminal_reason 버그 수정

이전엔 모든 아이템 적재 완료(`terminal_reason="no_items"`) 케이스가 `finished` 리스트에 빠져 있어 default penalty(-0.005)로 떨어지는 버그가 있었음. `finished` 판정에 `"no_items"` 추가하여 수정.

### 5-4. 종료 경로 rebuild/obs 스킵 + retry 한계 축소 + invalid reset 스킵

- **종료 경로 스킵**: `_finalize_step()`에서 `terminated=True`인 경우 `_safe_rebuild_candidates(check=True)`와 `build_obs(...)`를 모두 건너뛰고, 직전 `self._last_obs`를 그대로 반환. 근거: SB3 PPO는 `dones=True` 전이의 next obs를 사용하지 않으며, value bootstrap은 `info["TimeLimit.truncated"]`가 True일 때(=`truncated and not terminated`)만 발생하므로 truncated 경로는 그대로 obs를 빌드. `terminated`만 골라 스킵해도 안전.
  - 부산물: `__init__`/`reset()` 정상 경로에서 `self._last_obs`를 반드시 채워두도록 보강.
- **retry 한계 100→20**: `PalletFitEnv.__init__(max_retry_per_step=...)`의 기본값을 `100 → 20`으로 축소. 외부 override 호출처 없음. RETRY_LIMIT까지 도달하기 전 헛스텝의 상한이 1/5로 줄어 실패 연쇄가 빠르게 차단됨.
- **invalid reset 스킵**: `_build_items_from_plan`이 False를 리턴하는 invalid 에피소드 경로에서 `_safe_rebuild_candidates(check=True)`와 `build_obs(...)` 호출을 모두 제거. 새 헬퍼 `_make_zero_obs()`가 `observation_space` 형상에 맞는 0-채움 dict를 만들고, `_make_noop_only_state()`로 NOOP-only 마스크를 세팅. agent의 다음 step은 NOOP→즉시 종료로 끊기므로 진짜 obs/candidate는 필요 없음.

### 5-5. `Item.getDimension()` 캐싱 — `build_obs` 18× 단축

**진단 (cProfile, bin items=10, 50× build_obs)**:
- 이전: 4.20s 합계 = **84ms/call**, 함수 호출 6.5M회.
- 가장 두꺼운 호출: `_encode_item_from_obj` 2.62s (62%) → 내부 `get_direction_overlap` 2.42s.
- 진짜 원인: `Item.getDimension()`이 build_obs당 **27,100회** 호출됨 (~54×/아이템). 매 호출이 `is_axis_aligned` + 4× `np.allclose`를 돌려 결국 `numeric.isclose`를 19만 번 실행.

**수정**: `Item.__init__`에 `_dim_cache_key`/`_dim_cache_val` 추가, `getDimension()`은 `(width, height, depth, *rotation_quat)`를 키로 캐시 hit 시 즉시 반환.

```python
def __init__(...):
    ...
    self._dim_cache_key = None
    self._dim_cache_val = None

def getDimension(self):
    if self.name == 'gripper':
        return [self.width, self.height, self.depth]
    q = self.rotation_quat
    cache_key = (self.width, self.height, self.depth, q[0], q[1], q[2], q[3])
    if self._dim_cache_key == cache_key and self._dim_cache_val is not None:
        return list(self._dim_cache_val)   # 사본 반환 (caller mutation 안전)
    # ... 기존 axis-aligned / AABB fallback 로직 ...
    self._dim_cache_key = cache_key
    self._dim_cache_val = tuple(result)
    return result
```

**불변성 보장**:
- 외부에서 `it.rotation_quat = ...` 재할당 시 → 캐시 키가 자동으로 달라져 miss → 재계산.
- 외부에서 `it.width = ...` 등을 바꿔도 동일.
- 반환값은 매번 새 list(`list(cached_tuple)`) → caller의 mutation이 캐시를 오염시키지 않음.

**측정 (수정 후, 동일 시나리오)**:
- 4.20s → **0.234s** (cumulative profile) / wall-clock **2.60ms/call** = 약 **18× 단축**.
- 함수 호출 6.5M → 350K (95% 감소).
- 함께 빨라진 항목: `_encode_item_from_obj` 18×, `get_direction_overlap` 24×, `encode_globals` 24×, `get_score_Guillotine` 34× — 모두 내부에서 이웃 아이템들의 `getDimension()`을 반복 호출하던 경로.

**검증**:
- 캐시 hit 시 동일값 반환 ✓
- rotation_quat 변경 시 invalidate ✓
- 반환 list mutation이 다음 호출 결과에 영향 없음 ✓
- 전체 env step pipeline (reset → 5 successful steps → SU 계산) 정상 ✓

```python
# _finalize_step (요약)
if not terminated:
    self._safe_rebuild_candidates(check=True)
    head_q = queue_head(...)
    self._last_obs = build_obs(...)
# else: skip — 직전 _last_obs 재사용 (SB3가 사용 안 함)

# 성공/종료 시 GIF 캡처는 그대로 진행 (_bin만 사용, rebuild/obs와 무관)
```

---

## 6. 평가 병렬화 + GIF 자동 저장 (2026-04-27 적용 ✅)

### 6-1. 목표 (달성)

- 평가 wall time을 `n_envs_eval`× 단축.
- eval 경로를 `PalletFitEnv.step()`과 통일(중복 builder 호출 제거).
- 학습 진행 시각 확인용 render는 유지하되, **메인 프로세스가 bin 객체를 들지 않게** 0번 워커 내부에서 GIF로 저장.

### 6-2. 적용된 설계

#### (a) `make_eval_env` — Subproc 고정, 0번 워커만 GIF

```python
# agent.py
def make_single_env(seed, tb_log_dir, *, is_render_env=False):
    def _thunk():
        env = PalletFitEnv(seed=seed, tb_log_dir=tb_log_dir, is_render_env=is_render_env)
        return Monitor(env, info_keywords=info_keys)
    return _thunk

def make_eval_env(*, n_envs, base_seed, tb_log_dir):
    thunks = [
        make_single_env(base_seed + i, tb_log_dir, is_render_env=(i == 0))
        for i in range(n_envs)
    ]
    venv = SubprocVecEnv(thunks, start_method="spawn")
    return VecMonitor(venv, filename=str(Path(tb_log_dir) / "monitor_eval"))
```

- `backend="dummy"|"subproc"` 분기는 삭제. SubprocVecEnv 전용.
- `__init__`에서 `n_envs=cfg.n_envs_eval`로 생성.

#### (b) `PalletFitEnv` — 워커 내부 GIF 캡처

새 인자/상태:
```python
def __init__(self, *, seed, tb_log_dir=None, max_retry_per_step=100,
             is_render_env=False, gif_fps=4):
    ...
    self._is_render_env = is_render_env
    self._gif_fps       = gif_fps
    self._gif_capture_active = False
    self._gif_save_dir       = None
    self._gif_frames: list[np.ndarray] = []
    self._gif_episode_idx    = -1
```

활성화 규칙(`reset()`에서):
```python
consumed_pending = self._pending_plan is not None  # ★ 새 plan을 소비했을 때만
...
tag = item_payload.get("tag", "")
if (self._is_render_env
    and consumed_pending
    and tag.startswith("eval")
    and self._tb_log_dir is not None):
    self._gif_capture_active = True
    self._gif_save_dir       = self._tb_log_dir / "eval_renders"
    self._gif_frames         = []
    self._gif_episode_idx    = self._episode_idx
else:
    self._gif_capture_active = False
    self._gif_frames         = []
```

> `consumed_pending` 가드는 SB3 자동 재리셋(같은 plan 반복) 시 GIF 중복 저장을 막기 위한 것. agent의 라운드-로빈 사이에서 워커가 일찍 끝나면 자동 재리셋되는데, 이때 `_pending_plan`은 비어 있으므로 캡처가 자동으로 비활성화됨.

캡처 지점:
- `reset()` 마지막에 빈 bin 첫 프레임 1장.
- `_finalize_step()`에서 placement 성공 (`placed_item is not None`) 시마다 1장.
- 종료(`terminated or truncated`) + capture active일 때 `_flush_gif()`로 imageio v2 mimsave 후 비활성화.

저장 포맷:
- 경로: `{tb_log_dir}/eval_renders/ep{episode_idx:04d}_SU{su:.3f}.gif`
- 프레임 소스: `Bin.render(return_array=True, save=False, show=False)` → `(H,W,3) uint8`
- duration: `1/gif_fps`초 (기본 4 FPS)

#### (c) `evaluation()` — 라운드-로빈 vectorized 루프

기존 ~200줄 직렬 루프를 약 70줄로 축약:

```python
@th.no_grad()
def evaluation(self, episodes=None, max_eval_steps=200) -> Dict[str, float]:
    self.model.policy.set_training_mode(False)
    n_envs = self.eval_env.num_envs
    if episodes is None:
        episodes = n_envs

    # 1) plan 캐시 (재현성)
    if self._cached_eval_plans is None:
        self._cached_eval_plans = make_eval_plan_maker(self.cfg, ...)(0, episodes, current_steps=0)
    plans = self._cached_eval_plans
    if len(plans) < episodes:
        plans = list(plans) + [plans[-1]] * (episodes - len(plans))

    noop_idx = int(self.eval_env.get_attr("NOOP_IDX", indices=[0])[0])

    # 2) 라운드 단위로 plan 분배 → reset → batched predict → step
    su_list, packed_list = [], []
    plan_idx = 0
    while plan_idx < episodes:
        round_size = min(n_envs, episodes - plan_idx)
        for i in range(round_size):
            payload = env_plan_to_payload(plans[plan_idx + i])
            self.eval_env.env_method("apply_plan", payload, indices=[i])

        obs = self.eval_env.reset()
        done_mask = np.zeros(n_envs, dtype=bool)
        done_mask[round_size:] = True   # 미사용 워커는 done 취급

        for _ in range(max_eval_steps):
            if done_mask.all():
                break
            masks = get_action_masks(self.eval_env)
            actions, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
            actions = np.asarray(actions, dtype=np.int64)
            if done_mask.any():
                actions[done_mask] = noop_idx     # 끝난 워커는 NOOP 패딩
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

핵심 포인트:
- **plan 주입 → `vec_env.reset()`** 패턴: `apply_plan`이 `_pending_plan`만 세팅하고 reset에서 소비되는 기존 동작 그대로 활용.
- **마스크 조회**: `sb3_contrib.common.maskable.utils.get_action_masks(self.eval_env)` 사용 (내부에서 `env_method("action_masks")`).
- **NOOP 패딩**: 라운드 내 빠르게 끝난 워커가 자동 재리셋되어도 `_pending_plan`이 비어 있으므로 GIF 중복은 안 생김. NOOP을 보내 다음 step에서 즉시 종료시키며 슬로우 워커를 기다림.
- **의도적 제한**: `episodes < n_envs`인 경우, 사용 안 하는 워커들도 백그라운드에선 살아 있지만 `done_mask`로 무시됨.

#### (d) 삭제된 죽은 코드

- `evaluation()` 안의 직렬 builder 루프(`Packer` 직접 생성, `rebuild_candidates`/`build_obs`/`place_by_action` 직접 호출).
- `_last_eval_bins` 상태와 `render_last_eval()` 메서드 — 메인이 더 이상 bin 객체를 들지 않으므로 불필요.
- `PalletFitEnv.save_render()` PNG 헬퍼와 `_save_render_on_done`/`_save_render_dir` 분기 — `tag == "eval"` 정확매치라 plan_maker가 발급하는 `eval_offline`/`eval_online`/`eval_tsg`엔 매칭되지 않던 사실상 데드코드였음.
- `make_eval_env`의 `backend` 인자.
- `agent.py`에서 더 이상 쓰지 않는 imports: `DummyVecEnv`, `Packer`, `RotationType`, `deque`, `random`, `_generate_items_with_tsg`, `TSGConfig`, `global_item_manager`, `rebuild_candidates`, `place_by_action`, `build_observation_deque`, `queue_head`.

### 6-3. 검증 결과 (smoke test)

| 케이스 | 기대 동작 | 결과 |
|---|---|---|
| `is_render_env=True` + eval tag + NOOP 종료 | 1개 GIF 저장 | ✅ `eval_renders/ep0001_SU0.000.gif` |
| `is_render_env=False` + eval tag | GIF 미생성 | ✅ |
| `is_render_env=True` + 자동 재리셋(pending_plan 없음) | GIF 중복 미생성 | ✅ |

### 6-4. 예상 효과

| 평가 에피소드 수 | 이전 (Dummy 1, 직렬) | 현재 (Subproc N) | 속도 비 (≈) |
|---|---|---|---|
| `n_envs_eval` (=5) | t × 5 | max ≈ t | ~5× |
| `2 × n_envs_eval` | t × 10 | t × 2 | ~5× |

> Subproc 시작 비용은 학습 시작 1회로 amortize됨(eval_env는 재사용). 라운드 종료 대기 중 NOOP 패딩 step은 워커당 최대 1개 step 정도라 무시 수준.

---

## 7. 권장 적용 순서 (업데이트)

| 순서 | 항목 | 예상 효과 | 난이도 | 상태 |
|---|---|---|---|---|
| 1 | A. deepcopy 제거 | step당 수 ms 절감 | 쉬움 | ✅ 적용 |
| 2 | B. placed_item 전달 | stability/contact 활성화 | 쉬움 | ✅ 적용 |
| 3 | E. 조기 종료 deepcopy 회피 | 조기 종료 비용 제거 | 쉬움 | ✅ 적용 (A에 포함) |
| 4 | C. 종료 경로 rebuild/obs 스킵 | 종료 step 비용 제거 | 쉬움 | ✅ 적용 |
| 5 | D. retry 한계 축소 (100→20) | 실패 연쇄 차단 | 매우 쉬움 | ✅ 적용 |
| 6 | F+G. eval 병렬화 + evaluation 통합 + 0번 워커 GIF | 평가 ~5× 속도 | 중간 | ✅ 적용 |
| 7 | I. invalid reset rebuild/obs 스킵 | invalid 에피소드만 단축 | 매우 쉬움 | ✅ 적용 |
| 8 | H. 관측 빌드 완화 (Item.getDimension 캐싱) | build_obs ~18× | 쉬움 | ✅ 적용 |

---

## 8. 예상 성능 개선 누적

| 적용 항목 | step당 비용 변화 | 비고 |
|---|---|---|
| 기준 (이전 구조) | 100% | deepcopy + cand_item=None |
| ✅ + EDP dedup | check=True 구간 ~10× 감소 | (Task 1) |
| ✅ + A/B/E (deepcopy 제거) | ~70% | rollout time 1차 측정 권장 |
| ✅ + C (종료 스킵) | ~65% | terminated에선 rebuild/obs 비용 0 |
| ✅ + D (retry 20) | 실패 시나리오만 단축 | RETRY_LIMIT 도달까지의 헛스텝 1/5로 |
| ✅ + F+G (eval 병렬 + 0번 GIF) | eval wall time ~`n_envs_eval`× 감소 | rollout과 별개 |
| ✅ + H (Item.getDimension 캐싱) | build_obs 84ms → ~5ms/call (~18×) | _encode_item, encode_globals 모두 동시 가속 |

*다음 측정 포인트: 24 워커 × n_steps=128 = 3072 step 한 rollout의 wall time을 deepcopy/H 적용 전후로 비교. eval 시간은 `n_envs_eval`만큼 라운드 단축됐는지 콜백 로그(`time/eval_time` 등)로 확인.*
