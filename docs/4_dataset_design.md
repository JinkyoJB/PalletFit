# Dataset 설계 분석 및 개선 방향

> 대상 파일: `planning/RL/PalletFit_RL/agent.py` (Plan/Plan maker), `planning/RL/PalletFit_RL/env.py` (`_build_items_from_plan`), `planning/packer.py` (`_load_offline_data`/`_load_online_item_type_data`)
> 목적: 학습/평가용 데이터 소스 구조(`offline` / `online_type` / `tsg`)를 분석하고 의미·구현 양쪽의 미흡한 점을 정리해 개선 로드맵 제안.

---

## 1. 현재 구조 정리

### 1-1. 데이터 모드 3종

`EnvPlan.item_mode` 가 `"offline" | "online_type" | "tsg"` 중 하나:

| 모드 | 코드 위치 | 박스 출처 | 결정성 | 현재 용도 |
|---|---|---|---|---|
| **`offline`** | `Packer._load_offline_data` (packer.py:574) | JSON 파일의 박스 리스트를 그대로 로드 | 결정적(같은 파일=같은 박스) | "복습"용 고정 시나리오 |
| **`online_type`** | `Packer._load_online_item_type_data` (packer.py:589) | JSON에 정의된 "타입 거푸집"에서 random sampling, `max_items` 개수 생성 | 비결정적(seed에 따라 다름) | 다양성 확보 |
| **`tsg`** | `_generate_items_with_tsg` + `TrainsetGenerator` | bin 부피를 `init_slice`로 분할 → merge/split → SU=1.0 보장 시퀀스 절차적 생성 | 비결정적 | 이상적 reward 신호 학습 |

### 1-2. Plan 발급 흐름

```
[학습 시작]
  ↓
AgentConfig (phase 비율, rehearsal_p, online_type_ratio, tsg_ratio …)
  ↓
make_time_based_plan_maker(cfg, history)
  ↓ (rollout 시작 시 호출, n_envs개 plan 생성)
_make_plans_core(...)
  ↓
[per env i]
  - rehearsal_p 확률 → offline (rehearsal)
  - 아니면 p_scenario_fixed 확률 → offline (fixed)
  - 아니면 p_online_type vs p_tsg 비율 → online_type / tsg
  ↓
EnvPlan dict
  ↓ env_plan_to_payload(plan)
  ↓ env.env_method("apply_plan", payload, indices=[i])
  ↓
PalletFitEnv._build_items_from_plan(plan)
  - item_mode 분기 → packer 메서드 또는 _generate_items_with_tsg 호출
```

### 1-3. Phase 자동 조정 (`_phase_from_steps`)

| Phase | 시간 비율 | 시나리오 분포 (`_mix_ratio_from_phase`) | 현재 코드의 추가 강제 |
|---|---|---|---|
| `bootstrap` | 0 ~ `bootstrap_ratio` (10%) | 70 fixed : 30 random | `OVERFIT_TARGET_PATH` 1개로 강제 + preview_cnt=PREVIEW_MAX 강제 + rollout_idx=0 (시드 고정) |
| `mid` | `bootstrap_ratio` ~ `mid_ratio` (60%) | 50 fixed : 50 random | 정상 풀 사용 |
| `late` | 그 이후 | 40 fixed : 60 random | 정상 풀 사용 |

`fixed` = offline pool에서 file 1개 선택, `random` = online_type or tsg.

### 1-4. DatasetHistory 추적 항목

`SourceStat`: count, ema_su, last_su, last_ts. mode별 source_id로 grouping:
- offline: 파일 경로
- online_type: 파일 경로
- tsg: `"tsg"` (모두 한 그룹)

`mastered_offline()` 정의는 있지만 호출처 0건.

### 1-5. 평가 plan_maker (`make_eval_plan_maker`)

학습과 별도 로직: 5 envs 중 1 online + 4 offline 고정 패턴 (4:1 비율 hardcoded). offline pool은 `cfg.eval_item_paths_pool`.

---

## 2. 문제점 분석

### ~~2-1. ⚠️ 모드 이름이 의미를 왜곡~~ ✅ **해결 (2026-05-04)**

~~| 현재 이름 | 일반적 의미 (RL 문헌) | 실제 동작 | 충돌 |~~
~~|---|---|---|---|~~
~~| `offline` | 기록된 데이터로 학습 (RL agent의 학습 패러다임) | 미리 정의된 박스 시퀀스로 episode 진행 | **다른 도메인 단어와 혼동** |~~
~~| `online_type` | 실시간 데이터 스트림 | 박스 type 정의에서 random sample | "online"이 아님 |~~
~~| `tsg` | 약자 | TrainsetGenerator로 절차적 생성 | **외부인이 못 알아봄** |~~

**해결 — 옵션 C (behavioral)로 일괄 rename**:
- `offline` → **`recorded`** (미리 결정된 박스 시퀀스)
- `online_type` → **`type_sampled`** (타입 정의에서 sampling)
- `tsg` → **`synthetic`** (절차적 합성)

**변경 범위**:
- `agent.py`: `EnvPlan.item_mode` 기본값, 모듈 상수(`RECORDED_PATHS`, `TYPE_SAMPLED_PATHS`, `DEFAULT_SYNTHETIC_CFG`), AgentConfig 필드(`type_sampled_ratio`, `synthetic_ratio`, `recorded_paths_pool`, `recorded_paths_weights`, `recorded_auto_weight_by_len`, `eval_recorded_paths_pool`), 헬퍼 함수(`_resolve_recorded_pool_and_weights`, `_expand_recorded_pool_to_files`, `_choose_recorded_file_with_history`), `_make_plans_core` 인자(`p_type_sampled`, `p_synthetic`, `recorded_file_candidates`), `DatasetHistory` 필드(`recorded`/`type_sampled`/`synthetic`) + `mastered_recorded()`, plan tag(`eval_recorded`/`eval_type_sampled`/`eval_synthetic`).
- `env.py`: `_build_items_from_plan` 3 분기, `_build_binPacker_from_plan` 2 분기, `_as_plan_dict` default, `__main__` 디버그 러너의 `--mode` choices + payload key.
- `packer.py`: 메서드(`_load_recorded_data`/`_load_type_sampled_data`), 속성(`recorded_path`/`type_sampled_path`). 옛 kwarg(`offline_item_path`/`online_item_type_path`)는 backward-compat으로 1단계 유지.
- `experiments/get_test.py`: Packer 생성자 kwarg 새 이름으로 변경.
- payload key: `"offline_item_paths"` → `"recorded_paths"`, `"online_item_type_path"` → `"type_sampled_paths"`, `"tsg_cfg"` → `"synthetic_cfg"`.
- DatasetHistory CSV: 옛 mode 이름(offline/online_type/tsg)을 새 이름으로 자동 매핑하는 `_MODE_LEGACY_ALIAS` 추가 → 옛 `dataset_history.csv` 그대로 로드 가능.

**유지된 것**:
- `utils.util_functions.load_offline_data` 유틸 함수는 "디스크에서 로드"라는 일반 의미라 이름 유지 (10+ 호출처가 heuristics/experiments에 분산됨).
- env의 `tag.startswith("eval")` GIF 캡처 검사 — prefix가 동일하므로 자동 호환.

**검증**:
- `_make_plans_core(p_type_sampled=0.7, p_synthetic=0.3, recorded_file_candidates=...)` 정상 동작 ✓
- env recorded mode (queue len 150) ✓
- env synthetic mode (queue len 32) ✓
- DatasetHistory record / 옛 CSV load 호환 ✓
- TSG 32-item 정상 완주: ep_return=+82.22, SU=1.0, sum_terms == ep_return ✓
- env.py 디버그 러너(`--mode synthetic`) 정상 동작 ✓

### ~~2-2. ⚠️ 3 모드 분기가 코드 중복~~ ✅ **해결 (2026-05-04, DataSource 추상화)**

~~`env._build_items_from_plan`, `agent._make_plans_core`, `agent._build_binPacker_from_plan`, `DatasetHistory.record` — 4곳에서 모두 `if item_mode == "tsg" / elif "offline" / elif "online_type"` 분기. 새 데이터 소스 추가 시 4곳 수정 필요.~~

**해결 — DataSource Protocol/ABC 도입** (`planning/data/data_sources.py`):

| 위치 | 이전 | 현재 |
|---|---|---|
| `env._build_items_from_plan` | 3-way `if/elif` 분기 + 각 분기마다 packer 메서드 호출 | `make_source(spec).sample(seed)` 한 줄 |
| `env._build_binPacker_from_plan` | 2-way 분기로 bin alias 결정 | `plan["bin"]` 우선 + `source.bin_alias_hint()` fallback |
| `agent._make_plans_core` | `_make_base_plan(item_mode, payload, tag)` 헬퍼 + 분기 | `spec_recorded/spec_type_sampled/spec_synthetic` 헬퍼 + plan에 `source` 필드 단일 |
| `agent.DatasetHistory.record` | mode별 source_id 결정 분기 | `record_from_info(info)` — env가 info에 노출한 source_mode/source_id 직접 사용 |

**새 source 추가 비용**:
- 이전: 5곳 수정 (env 분기, plan_maker 분기, packer 메서드, history 분기, eval_plan_maker 분기)
- 현재: **DataSource subclass 1개 작성 + `register_source(klass)` 호출** = 2 line만

**EnvPlan 변경**:
```python
@dataclass
class EnvPlan:
    seed: int
    source: Dict[str, Any]   # ← {"mode": ..., "args": ...} 단일 spec
    bin_alias: str
    ...
    # (제거: item_mode, item_payload)
```

**검증 결과**:
- 3 mode 모두 `make_source(spec).sample(seed)` 단일 호출로 박스 생성 ✓
- env step pipeline 정합 (sum_terms == ep_return) 보존 ✓
- info에 `source_mode`/`source_id` 자동 노출 → history 통계 자동 적립 ✓

### ~~2-3. ⚠️ bin alias가 data mode에 의존 (의미적 결합)~~ ✅ **해결 (2026-05-04, DataSource 추상화)**

~~- data mode가 bin 종류를 결정 → **bin과 data가 강하게 coupling**.~~
~~- 같은 박스 데이터를 다른 bin 크기로 실험하고 싶어도 mode를 바꿔야 함.~~

**해결**: bin alias 결정이 plan-주도로 변경:
1. `plan["bin"]` 명시되면 그대로 사용 (1순위)
2. 미지정이면 `source.bin_alias_hint()` (2순위, source가 선호 bin 표명 가능)
3. 둘 다 없으면 `"experiment_RL"` fallback

→ 같은 박스 데이터를 다른 bin에서 실험 가능. coupling 끊김.
→ `bin_alias_hint`는 옵션 (RecordedSource→`"experiment_RL"`, TypeSampledSource→`"default2"` 옛 동작 보존). 강제 X, plan에서 override 가능.

### ~~2-4. ⚠️ "rehearsal" 개념과 구현 어긋남~~ ✅ **해결 (2026-05-04, 선택 B — 진짜 rehearsal)**

~~`_choose_offline_file_with_history`는 "안 배운 파일 우선" 로직 → 이름과 동작 충돌. mastered_offline() 메서드는 정의만 있고 호출 0.~~

**해결** — 함수 분리 + 분기별로 다른 함수 호출:

```python
# 새 헬퍼 (agent.py)
def _choose_unseen_first(files, weights, history):
    """안 배운 source 우선 (탐험). 평소 학습용."""
    unseen = [f for f in files if history.recorded.get(f).count == 0]
    return random.choice(unseen) if unseen else _weighted_choice(files, weights)

def _choose_mastered_first(files, weights, history):
    """이미 mastered한 source 우선 (catastrophic forgetting 방지 = 진짜 rehearsal)."""
    mastered = [f for f in files if f in history.mastered_recorded()]
    return random.choice(mastered) if mastered else _weighted_choice(files, weights)

# _make_plans_core 분기
if np.random.rand() < rehearsal_p:                       # ← 진짜 복습
    chosen = _choose_mastered_first(...)
elif np.random.rand() < p_scenario_fixed:                # ← 평소 학습 (탐험)
    chosen = _choose_unseen_first(...)
```

**선택 B 채택 근거**:
- 이름과 동작 일치 (`rehearsal_p` → mastered 우선 sample).
- `mastered_recorded()` 메서드가 진짜 호출됨 (이전 dead code).
- 학습 초기 fallback: mastered=∅이면 일반 sample → 학습 초기엔 fixed 분기와 동일 동작 → 안전.

**임계값** (`DatasetHistory`): 그대로 유지.
- `min_episodes_master = 10` (10번 이상 등장)
- `su_master_threshold = 0.85` (ema_su ≥ 0.85)

**검증** (4가지 시나리오):
- 학습 초기 (mastered=∅): mastered_first fallback → 균등 sample (A 7, B 8, C 5) ✓
- A만 mastered: rehearsal 분기 100% A에 집중 (20/20) ✓
- 같은 history에서 unseen_first: C(unseen) 100% 집중 (20/20) ✓
- plan_maker 통합: rehearsal 12개 모두 A(mastered) / fixed 5개 모두 B·C(unseen) ✓

### ~~2-5. ⚠️ Phase 강제 vs Config 사용자 지정 충돌~~ ✅ **해결 (2026-05-04, BootstrapPolicy 도입)**

~~- `cfg.online_type_ratio` / `cfg.tsg_ratio`를 사용자가 설정해도, `bootstrap` phase에선 무조건 `OVERFIT_TARGET_PATH` 1개로 덮어씀.~~

**해결** — `BootstrapPolicy` dataclass로 강제 항목들을 명시화:

```python
@dataclass
class BootstrapPolicy:
    enabled: bool = True
    duration_ratio: float = 0.10                            # 옛 cfg.bootstrap_ratio
    fixed_recorded_path: Optional[str] = OVERFIT_TARGET_PATH
    fixed_preview_cnt: Optional[int] = None                 # None=PREVIEW_MAX
    fixed_rollout_seed: bool = True
    force_recorded_only: bool = True

@dataclass
class AgentConfig:
    bootstrap: BootstrapPolicy = field(default_factory=BootstrapPolicy)
    # bootstrap_ratio 필드 제거 (BootstrapPolicy.duration_ratio로 이전)
    ...
```

`make_time_based_plan_maker`의 bootstrap 분기:
```python
bs = cfg.bootstrap
if bs.enabled and phase == "bootstrap":
    if bs.fixed_recorded_path:    recorded_files = [bs.fixed_recorded_path]
    if bs.force_recorded_only:    real_p_fixed, real_p_type_sampled, real_p_synthetic = 1.0, 0.0, 0.0
    if bs.fixed_rollout_seed:     final_rollout_idx = 0
    if bs.fixed_preview_cnt is not None:  cur_pc_choices = [bs.fixed_preview_cnt]
    else:                                  cur_pc_choices = [PREVIEW_MAX]
```

**기본값은 옛 동작과 동일** — 학습 동작 무변화. cfg를 명시적으로 바꿀 때만 다르게 동작.

**사용 예**:
```python
# bootstrap 완전 끄기
cfg = AgentConfig(bootstrap=BootstrapPolicy(enabled=False))

# 다른 OVERFIT 파일로 강제
cfg = AgentConfig(bootstrap=BootstrapPolicy(fixed_recorded_path="my.json"))

# bootstrap에서도 type_sampled 허용
cfg = AgentConfig(bootstrap=BootstrapPolicy(force_recorded_only=False))

# bootstrap을 짧게 (5%)
cfg = AgentConfig(bootstrap=BootstrapPolicy(duration_ratio=0.05))
```

**검증** (7가지 시나리오):
- 기본값 = 옛 동작 (OVERFIT_TARGET_PATH 강제, recorded만, preview=PREVIEW_MAX, rollout_idx=0) ✓
- `enabled=False` → bootstrap 무시, cfg 비율 정상 적용 (recorded 12, type_sampled 5, synthetic 3) ✓
- `fixed_recorded_path=다른 파일` → 사용자 지정 파일로 강제 ✓
- `force_recorded_only=False` → bootstrap에서도 type_sampled 등장 ✓
- 페이즈 판정(`_phase_from_steps`)은 `bootstrap.duration_ratio` 사용 ✓
- mid/late phase → bootstrap 적용 안 됨, 정상 풀 사용 ✓

→ docs/4 Sec 3 옵션 I (Bootstrap policy 명시화)와 J (OVERFIT_TARGET_PATH → Config) 두 항목 + Sec 2-12 모두 한 번에 해결.

### ~~2-6. ⚠️ TSG cfg가 bin 크기와 무관하게 hardcoded~~ ✅ **완전 해결 (2026-05-05, B+C 결합)**

#### 무엇이 문제였나

`DEFAULT_SYNTHETIC_CFG.bin_size = (1000, 1000, 1000)` 기본값이 박혀 있어서, 사용자가 `bin_alias = "default2"` (실제 500×500×400)를 쓰면서 synthetic_cfg를 안 바꾸면:
- SyntheticSource가 1m³짜리 박스를 만듦
- env가 default2 (0.5×0.5×0.4)에 그 박스들을 넣으려고 시도 → **첫 박스부터 bin보다 커서 즉시 실패**

#### 어떻게 해결했나 (B + C 결합)

**B: SyntheticSource가 bin_alias 인자 받아 BIN_SPECS lookup**:
```python
class SyntheticSource:
    def __init__(self, bin_alias=None, **cfg_kwargs):
        if bin_alias and "bin_size" not in cfg_kwargs:
            spec = BIN_SPECS.get(bin_alias)
            if spec:
                cfg_kwargs["bin_size"] = (spec["width"], spec["height"], spec["depth"])
        ...

# 사용: spec_synthetic(bin_alias="default2") → 자동 (500, 500, 400)
spec = spec_synthetic(bin_alias="default2", max_items=20)
```

**C: env가 plan["bin"]을 source에 자동 주입 (안전망)**:
```python
# env._build_items_from_plan
spec = plan.get("source") or {}
if spec.get("mode") == "synthetic":
    args = dict(spec.get("args", {}))
    if "bin_size" not in args and "bin_alias" not in args:
        bin_alias = plan.get("bin")
        if bin_alias:
            args["bin_alias"] = bin_alias
            spec = {**spec, "args": args}
```

→ 사용자가 spec에 bin_alias를 명시 안 해도, env가 plan["bin"]을 자동 주입해 동기화.

#### 우선순위 (명시적 override가 항상 우선)

```
1순위: spec.args["bin_size"]   (사용자가 cfg에 명시한 값)
2순위: spec.args["bin_alias"] → BIN_SPECS lookup  (사용자가 명시한 alias)
3순위: plan["bin"] → BIN_SPECS lookup            (env 자동 주입, 안전망)
4순위: SyntheticConfig 기본값 (1m³)              (다 없을 때)
```

→ 사용자 명시가 항상 우선. env 자동 주입은 사용자 잊어도 안전한 fallback.

#### 사용 예

```python
# (1) 그냥 plan["bin"]만 명시 — env가 자동 동기화 (가장 일반적)
plan = EnvPlan(seed=0, source=spec_synthetic(max_items=30), bin_alias="default2")
# → SyntheticSource가 default2 (500x500x400)에 맞는 박스 생성

# (2) spec에 bin_alias 명시 — 의도 명확
plan = EnvPlan(seed=0, source=spec_synthetic(bin_alias="default2", max_items=30),
               bin_alias="default2")

# (3) bin_size 직접 override (특수 시나리오)
plan = EnvPlan(seed=0, source=spec_synthetic(bin_size=(300, 300, 300), max_items=20),
               bin_alias="default2")
# → source는 (300,300,300) 박스 생성, bin은 default2 (500x500x400) — 의도적 mismatch
```

#### 검증 (5 시나리오 모두 통과)

| 시나리오 | bin_size | 결과 |
|---|---|---|
| spec.bin_size 명시 + bin_alias도 명시 | spec 우선 | (700, 700, 700) ✓ |
| spec.bin_alias만 명시 (default2) | BIN_SPECS lookup | (500, 500, 400) ✓ |
| 둘 다 미지정 | 기본값 1m³ | (1000, 1000, 1000) ✓ |
| env 자동 주입 (plan.bin=default2, source.args 미명시) | plan.bin → BIN_SPECS | (500, 500, 400) ✓ |
| 사용자 spec.bin_size override + plan.bin 다름 | spec 우선 (의도적 mismatch 허용) | (300, 300, 300) ✓ |

#### 코드 변경량

| 파일 | 변경 |
|---|---|
| `data_sources.py` SyntheticSource | bin_alias 인자 추가 (~6줄) |
| `data_sources.py` spec_synthetic | bin_alias 인자 + args에 포함 (~3줄) |
| `env.py _build_items_from_plan` | synthetic source plan["bin"] 자동 주입 (~7줄) |
| **합계** | **~16줄** |

### ~~2-7. ⚠️ DatasetHistory의 source_id 입자가 mode마다 다름~~ ✅ **해결 (2026-05-04, DataSource 추상화)**

~~- offline/online_type: 파일 경로 단위 → 파일별 SU 추적 가능~~
~~- tsg: 모두 `"tsg"` 한 그룹 → init_slice 차이 추적 불가~~

**해결**: source_id 결정이 각 DataSource 클래스의 책임이 됨.
- `RecordedSource`: `"recorded:<chosen_path>"` (선택된 파일 경로)
- `TypeSampledSource`: `"type_sampled:<chosen_path>"`
- `SyntheticSource`: `"synthetic:n=30_min=100_asp=3.0"` ← **cfg variant 별 분리**

→ synthetic도 cfg마다 별개 통계 적립 → curriculum 전환 시 어떤 cfg에서 SU가 안 오르는지 추적 가능.

### ~~2-8. ⚠️ eval/train plan_maker 분리~~ ✅ **해결 (2026-05-05, EvalPolicy + 함수 통합)**

#### 무엇이 문제였나

train과 eval이 **다른 박스 분포에서 SU를 측정**했음. 이게 왜 문제냐면:

> 학습 중 TB 그래프에서 eval SU가 안 오르고 있다. 이유가 무엇일까?
> - (a) 정책이 아직 학습이 부족
> - (b) 학습한 분포(train)와 평가 분포(eval)가 달라서 transfer 안 됨
>
> → 두 분포가 다르면 (a)와 (b)를 **구별할 수 없음**.

구체적으로 어떻게 달랐는지:

| 항목 | train | eval |
|---|---|---|
| 박스 mode 비율 | phase + cfg 비율 | hardcoded **4 recorded : 1 type_sampled** (`(i+1) % 5 == 0`) |
| margin | random `(0~8)` | 0 고정 |
| preview_cnt | random `[1~5]` | `PREVIEW_MAX` 단일 |
| rehearsal/explore | 있음 | 없음 |
| seed offset | `cfg.seed + rollout*10000` | `cfg.seed + 100_000 + rollout*10000` |
| 코드 경로 | `_make_plans_core(...)` (메인) | **별도 200줄 직렬 loop** |

→ "eval은 별도 함수에서 별도 로직"이라 train 어떤 변경이 있어도 eval은 동기화 안 됨. 새 mode 추가 시 두 함수 다 수정 필요.

#### 어떻게 해결했나 (3 단계)

**Step 1 — 분포를 dataclass로 명시**: `EvalPolicy` 도입. eval에서 어떤 분포를 쓰는지 cfg 한 곳에서 한눈에 보임.

```python
@dataclass
class EvalPolicy:
    # 1) source 분포 (default = train mid phase와 동일 → 비교 가능)
    p_scenario_fixed: float = 0.5
    p_type_sampled:   float = 1.0
    p_synthetic:      float = 0.0
    rehearsal_p:      float = 0.0    # eval은 rehearsal 안 함

    # 2) deterministic 옵션 (재현성)
    margin_x: int = 0
    margin_y: int = 0
    preview_cnt: Optional[int] = None  # None=max(cfg.preview_cnt_choices)
    bin_alias: str = "experiment_RL"

    # 3) eval 전용 데이터 (None이면 cfg fallback)
    recorded_pool: Optional[List[str]] = None
    type_sampled_paths: Optional[List[str]] = None

    # 4) seed 분리 (학습 시드 충돌 방지)
    seed_offset: int = 100_000
```

**Step 2 — 같은 코드 경로 사용**: `make_eval_plan_maker`가 자체 로직을 만들지 않고 train의 `_make_plans_core`를 그대로 호출. 차이는 인자로만:
- `history=None` → rehearsal/explore 비활성 (deterministic)
- `margin_range=(m, m+1)` → 단일값 강제
- `preview_cnt_choices=[ep.preview_cnt]` → 단일값 강제
- seed offset 분리

```python
def make_eval_plan_maker(cfg):
    ep = cfg.eval
    # ...pool/preview 결정...

    def plan_maker_eval(rollout_idx, n_envs, *, current_steps):
        # train과 동일 함수 호출!
        plans = _make_plans_core(
            cfg.seed + ep.seed_offset, n_envs,
            p_scenario_fixed=ep.p_scenario_fixed,
            p_type_sampled=ep.p_type_sampled,
            p_synthetic=ep.p_synthetic,
            rehearsal_p=ep.rehearsal_p,
            rollout_idx=rollout_idx,
            bin_key=ep.bin_alias,
            margin_range=(ep.margin_x, ep.margin_x + 1),
            recorded_file_candidates=recorded_files,
            recorded_file_weights=None,
            history=None,
            preview_cnt_choices=[preview_cnt_eval],
        )
        # tag만 eval_* 로 재라벨 (분석/로그/GIF용)
        for p in plans:
            p.mode = f"eval_{p.source['mode']}"
        return plans
    return plan_maker_eval
```

**Step 3 — 호출 측 단순화**: 이전엔 함수 인자 9개를 채워야 했음. 이제 `cfg` 하나만:

```python
# 이전
plan_maker = make_eval_plan_maker(
    self.cfg,
    eval_recorded_pool=self.cfg.eval_recorded_paths_pool,
    # eval_offline_weights, allow_online_type, allow_tsg, offline_path,
    # online_type_path, tsg_cfg, bin_key 모두 default
)

# 이후
plan_maker = make_eval_plan_maker(self.cfg)
```

#### 무엇이 좋아졌나

**(1) train/eval SU 직접 비교 가능**:
- default가 train mid 분포와 동일 → 두 곡선이 같은 좌표계 위에 있음.
- 측정 검증: 동일 시드에서 train mid (50 plans: recorded 22, type_sampled 28) vs eval (50 plans: recorded 25, type_sampled 25) → 거의 동일 분포 ✓

**(2) 한 곳만 수정하면 양쪽에 적용**:
- 새 source 추가 시: `_make_plans_core` 한 곳만 수정 → train/eval 자동 동기화.
- 이전엔 두 함수 모두 수정해야 했음.

**(3) eval 정책이 cfg에 명시됨**:
- "eval에서 어떤 분포로 평가하는지" cfg.eval 한 곳만 보면 됨.
- 옛날엔 함수 코드 + 모듈 상수 + cfg 세 곳을 봐야 했음.

**(4) deterministic 보장**:
- `seed_offset`으로 학습 시드와 분리.
- `_cached_eval_plans` 캐싱과 함께 매 평가마다 같은 plan 사용 → SU 비교 안정.

**(5) 사용자 override 자유도 ↑**:
```python
# eval에서도 type_sampled 절반, synthetic 반반
cfg = AgentConfig(eval=EvalPolicy(p_type_sampled=0.5, p_synthetic=0.5))

# eval에서도 다양한 margin으로 평가
cfg = AgentConfig(eval=EvalPolicy(margin_x=4, margin_y=4))

# eval만 큰 박스로 (curriculum stress test)
cfg = AgentConfig(eval=EvalPolicy(
    p_scenario_fixed=0.0, p_type_sampled=0.0, p_synthetic=1.0,
))
```

#### 코드 양 비교

| 항목 | 이전 | 이후 |
|---|---|---|
| `make_eval_plan_maker` 본체 | ~70줄 (시그니처 9개 파라미터, hardcoded loop, EnvPlan 직접 생성) | ~30줄 (`_make_plans_core` 위임) |
| `EvalPolicy` 신규 | 0 | ~25줄 |
| 호출 측 | 4-5줄 (인자 펼침) | 1줄 |
| **순 변화** | | **약 −20줄** + 구조 응집 ↑ |

라인 수보다 더 큰 효과는 **"두 코드 경로 동기화 부담 해소"** 와 **"eval 분포가 한 곳에서 명시"**.

#### 검증

| 시나리오 | 결과 |
|---|---|
| `EvalPolicy()` default → 5 plans 생성, mode 라벨 `eval_recorded`/`eval_type_sampled` 정확 | ✓ |
| Deterministic: 같은 seed 두 번 호출 → 동일 plan (seeds=[101456..101460]) | ✓ |
| Train mid (50 plans) vs Eval (50 plans) 분포 거의 동일 (recorded 22 vs 25, type_sampled 28 vs 25) | ✓ |
| `EvalPolicy(p_synthetic=1.0)` → eval 전부 synthetic | ✓ |
| `EvalPolicy(margin_x=4, margin_y=4)` → 모든 plan에 margin (4, 4) | ✓ |

→ docs/4 Sec 5 옵션 F (eval/train plan_maker 통합) 해결.

### ~~2-9. ⚠️ Mode 간 박스 분포 차이가 너무 큼~~ ⚠️ **부분 해결 (2026-05-04)**

~~- **TSG** `init_slice=(4,4,2)`: 32개, 모두 250mm급, 균일.~~
~~- **Online** `real_box.json`: 100~500mm, 다양한 비대칭.~~
~~- **Offline** `paper/testset`: 미리 결정된 시퀀스, 시드별 다른 구성.~~

- **synthetic** (옛 TSG): 옛날엔 `init_slice=(4,4,2)`라 모든 박스 250mm급 균일. 이제 새 generator는 aspect ratio ≤ 3.0 + min_dim ≥ 100mm 제약 안에서 다양한 크기(120~700mm 분포). type_sampled/recorded와 박스 크기 분포가 가까워짐 (Sec 5 doc 4-2 표 참조).
- ⚠️ **여전히 미세한 차이**: type_sampled의 type 정의는 외부 데이터 → 더 다양함. recorded는 미리 정해진 시퀀스 → 시드별 분포 다름. **완전 해소는 아님**, 다만 synthetic이 너무 균일하던 문제는 해결됨.

**2026-05-05 추가 결정 — 옵션 (e) 채택: 학습 결과 보고 결정**

사용자 의도: 큰 네트워크(`ft_d_model=256, ft_depth=6, ft_n_heads=8` 등)로 다양한 분포에 강건한 agent 학습.
이 의도가 자동으로 작동하는지는 학습 곡선으로 확인:

**관찰 포인트 (TB metric)**:
- `train/value_loss`: mode 간 큰 차이가 있는지. 차이가 크면 critic이 mode를 못 구별하고 평균 V(s)에 fitting.
- `train/explained_variance`: 0.7 미만으로 떨어지면 V(s) fitting 실패 신호.
- `train/{recorded,type_sampled,synthetic}_SU`: mode별 SU. 한 mode만 안 오르면 그 mode가 학습 안 됨.
- `eval/SU` vs `train/recorded_SU`: gap이 0.1 이상 벌어지면 train/eval transfer 실패 (Sec 2-8 EvalPolicy로 분포는 align했으니 큰 gap이면 진짜 OOD 문제).

**문제 보이면 도입할 옵션**:
- **옵션 (b) Mode embedding**: `globals` 6차원 → 9차원으로 확장 (mode one-hot 3 추가). critic이 mode를 즉시 인식 → V(s)가 mode-conditional하게 학습. 코드 변경 ~10줄. 단 input dim 변경이라 모델 재학습 필요.
- **옵션 (d) Domain randomization**: synthetic의 `min_item_mm`/`max_aspect_ratio`/`max_items` 등을 episode마다 랜덤 → 각 mode 안에서도 분포 다양 → mode 간 차이 흡수.

**왜 지금 변경 안 함**:
- 큰 네트워크가 분포 차이를 흡수할 capacity는 충분 (이론적으로).
- 다만 obs에 mode 정보가 명시 X → critic이 implicit으로 mode 추론해야 함 → 학습 초반 분산 클 수 있음.
- 추측 기반 변경보다 학습 결과(TB metric) 보고 데이터 기반 결정이 안전.

### ~~2-10. ⚠️ 새 데이터 소스 추가가 어려움~~ ✅ **해결 (2026-05-04, DataSource 추상화)**

~~가령 "실측 carton 박스 데이터셋"을 추가하고 싶으면 5곳 수정 필요.~~

**해결** — 새 source 추가 비용:

```python
# planning/data/data_sources.py에 클래스 1개 추가
class CartonSource(DataSource):
    mode = "carton"
    def __init__(self, paths: List[str]):
        self.paths = paths
        self._last = None
    def sample(self, seed):
        rng = random.Random(seed)
        self._last = rng.choice(self.paths)
        return load_carton_data(self._last)
    def source_id(self): return f"carton:{self._last}"
    def bin_alias_hint(self): return None

# Registry 등록 (1줄)
register_source(CartonSource)

# Plan_maker에서 사용 (헬퍼 1개 추가)
def spec_carton(paths): return {"mode": "carton", "args": {"paths": paths}}
```

→ env, packer, history는 손대지 않아도 됨. **2 line으로 새 source 추가 가능**.

### ~~2-11. ⚠️ `type_sampled`의 random weight 재배정 동작이 sloppy~~ ✅ **해결 (2026-05-05, uniform default + 사용자 override)**

#### 무엇이 문제였나

옛 `_sample_from_templates`:
```python
weights = [rng.randint(1, 10) for _ in range(len(templates))]   # ⚠️ 매 호출 random 재배정
for i in range(n_items):
    idx = rng.choices(range(len(templates)), weights=weights)[0]
```

→ 같은 type 파일을 줘도 episode마다 weights가 임의로 달라짐. seed 고정하면 재현 가능하긴 했지만 **"왜 이런 weights?"** 라는 의미가 불명. 사용자가 분포를 통제할 수단도 없음.

#### 어떻게 해결했나

`weights` 인자를 명시적으로 받게 변경. **default는 uniform** (모든 type 동일 확률).

```python
# planning/data/data_sources.py
def _sample_from_templates(templates, n_items, rng, weights=None):
    if weights is None:
        weights = [1.0] * len(templates)         # uniform default
    elif len(weights) != len(templates):
        raise ValueError(...)                    # 길이 검증
    for i in range(n_items):
        idx = rng.choices(range(len(templates)), weights=weights)[0]
        ...

class TypeSampledSource(DataSource):
    def __init__(self, paths, n_items=30, weights=None):
        # weights=None → uniform / list → 사용자 지정
        self.weights = weights

def spec_type_sampled(paths, n_items=30, weights=None):
    args = {"paths": paths, "n_items": n_items}
    if weights is not None: args["weights"] = list(weights)
    return {"mode": "type_sampled", "args": args}
```

옛 random 재배정 동작은 **완전 제거** (의미 불명이었음).

#### 사용 예

```python
# default — uniform 분포 (재현성 ↑, 통계 평가 가능)
spec = spec_type_sampled(paths=["real_box.json"], n_items=30)

# 큰 박스 위주로 학습/평가 (B type을 5배)
spec = spec_type_sampled(paths=["real_box.json"], n_items=30, weights=[1, 5, 1, 1, ...])

# 한 type만 100% (특정 type stress test)
spec = spec_type_sampled(paths=["real_box.json"], n_items=30, weights=[0, 1, 0, 0, ...])
```

#### 무엇이 좋아졌나

| 측면 | 이전 | 이후 |
|---|---|---|
| 매 episode 분포 | random 재배정 (의미 불명) | **uniform 또는 사용자 지정** (의미 명확) |
| 재현성 | seed 고정 시만 | **항상 deterministic** (default uniform) |
| 사용자 통제 | 불가 (코드 수정해야) | `weights=[...]`로 분포 명시 |
| 통계 평가 | 어려움 (분포 매번 변함) | 쉬움 (분포 안정) |
| 길이 mismatch | 조용히 진행 | **ValueError로 즉시 알림** |

#### 검증

- **재현성**: 같은 seed 두 번 호출 → 완전 동일 분포 (`{box8: 10, SLAMTEX: 10, box4u211: 9, ...}` 양쪽 동일) ✓
- **uniform default**: 1000 sample, 3 type → 약 333씩 ({A: 329, B: 336, C: 335}) ✓
- **weights override**: `[1, 5, 1]` → 약 1:5:1 비율 (154:703:143) ✓
- **mismatch validation**: weights 길이(4) ≠ templates 길이(38) → ValueError 즉시 발생 ✓

### ~~2-12. ⚠️ `OVERFIT_TARGET_PATH` 하드코딩~~ ✅ **해결 (2026-05-04, BootstrapPolicy 도입과 함께)**

~~- 모듈 최상단 상수로 박혀 있음.~~

- 모듈 상수 자체는 보존(default 출처)하되, `BootstrapPolicy.fixed_recorded_path`의 default value로 흡수됨.
- 사용자가 `cfg.bootstrap.fixed_recorded_path = "..."`로 override 가능 → "왜 이 파일인지" 의도가 cfg 레벨에서 명시됨.

---

## 3. 개선 제안 (우선순위 순)

### A. **모드 명명 재정의** (low risk, high readability)

| 현재 | 제안 | 이유 |
|---|---|---|
| `offline` | `dataset` 또는 `recorded` | 미리 결정된 박스 시퀀스 |
| `online_type` | `template` 또는 `type_sampled` | 타입 정의에서 sampling |
| `tsg` | `procedural` 또는 `generated` | 알고리즘으로 합성 |

문서/코드/log 모두 통일. backward-compat은 alias 매핑으로 1단계 유지 가능 (`offline` → `dataset` warning 후 deprecate).

### B. **DataSource 추상화** (medium risk, high extensibility)

3 mode 분기 4곳을 단일 Protocol/ABC로 통합:

```python
from typing import Protocol

class DataSource(Protocol):
    """Episode당 박스 리스트 + 메타데이터를 제공하는 인터페이스."""
    def sample(self, seed: int) -> List[Item]: ...
    def source_id(self) -> str: ...        # DatasetHistory 키
    def difficulty_hint(self) -> dict: ... # 학습 난이도 메타 (옵션)

class FileDataset(DataSource):
    """미리 정의된 박스 시퀀스 (현 offline)."""
    def __init__(self, path: str): ...
    def sample(self, seed): return load_offline_data(self.path)
    def source_id(self): return f"file:{self.path}"

class TemplateSampler(DataSource):
    """타입 거푸집 + random sampling (현 online_type)."""
    def __init__(self, type_path: str, n_items: int): ...

class ProceduralGenerator(DataSource):
    """알고리즘 생성 (현 tsg)."""
    def __init__(self, cfg: TSGConfig): ...
    def source_id(self): return f"tsg:slice={self.cfg.init_slice}"  # init_slice 반영

# Plan에서 DataSource 인스턴스 참조 (또는 factory key)
@dataclass
class EnvPlan:
    seed: int
    source: DataSource | str   # str이면 registry에서 lookup
    bin_alias: str
    ...

# env._build_items_from_plan
def _build_items_from_plan(self, plan):
    items = plan.source.sample(plan.seed)
    self.queue = deque(it._id for it in items)
```

→ 3 mode 분기 4곳이 1곳으로 축소. 새 data source 추가 시 1 클래스만 작성.

### C. **bin과 dataset 분리** (low risk, high flexibility)

```python
# 변경: env._build_binPacker_from_plan
def _build_binPacker_from_plan(self, plan):
    bin_alias = plan["bin"]   # 항상 plan에서 명시 받음, mode 의존 제거
    ...
```

- plan_maker가 bin alias를 명시적으로 결정.
- 같은 데이터를 다른 bin 크기로 학습 가능 → 실험 변수 늘어남.
- 의미적 coupling 제거.

### D. **TSG cfg 자동 동기화** (low risk, correctness)

```python
def auto_tsg_cfg(bin_alias: str) -> TSGConfig:
    spec = BIN_SPECS[bin_alias]
    return TSGConfig(
        bin_size=(spec.width, spec.height, spec.depth),
        init_slice=_init_slice_from_size(spec),  # 크기 비례 자동
        min_item_mm=100,
    )
```

bin_size 미스매치 위험 제거. cfg를 명시 지정도 여전히 가능.

### E. **DatasetHistory 활용 — Difficulty-weighted curriculum** (medium risk, real benefit)

현재 history는 통계만 쌓고 학습에 영향 안 미침. 활용:

```python
# plan_maker 안에서
def _difficulty_weighted_choice(sources, history, alpha=2.0):
    """잘 못하는(SU 낮은) source의 sampling 확률 ↑."""
    weights = []
    for s in sources:
        st = history.get(s.source_id())
        # ema_su가 낮을수록 가중치 ↑ (역수 + α 매개변수)
        w = 1.0 / max(0.1, st.ema_su) ** alpha if st.count > 0 else 1.0
        weights.append(w)
    return _sample_with_weights(sources, weights)
```

- 어려운 source를 더 자주 학습 → curriculum 자연 형성.
- `mastered_offline()` 활용해 마스터한 것은 sampling 빈도 낮춤.
- 진짜 rehearsal: 마스터한 것도 가끔 등장(예: 10% 확률) → catastrophic forgetting 방지.

### F. **eval/train plan_maker 통합** (medium risk, high diagnostic value)

```python
def make_unified_plan_maker(cfg, history, *, mode_filter=None):
    """train과 eval이 같은 로직 사용. mode_filter로 변형."""
    def maker(rollout_idx, n_envs, *, current_steps):
        # 공통: source 선택, bin 선택, preview_cnt 등
        # 차이: mode_filter='train'이면 random margin/preview, 'eval'이면 고정
        ...
```

- train SU와 eval SU가 같은 분포에서 측정됨 → 직접 비교 가능.
- 이미 `_make_plans_core`가 거의 그 역할 — 호출 인자로 train/eval 차이만 표현.
- 현재 `make_eval_plan_maker`의 hardcoded 4:1 패턴 → cfg.eval_source_ratio 같은 파라미터화.

### G. **rehearsal 정리** (low risk, semantic correctness)

선택 1: **이름을 실제 동작에 맞춤**.
```python
explore_p_boot, explore_p_mid, explore_p_late  # "안 배운 것 우선 탐색"
```

선택 2: **진짜 rehearsal로 동작 변경**.
```python
def _choose_for_rehearsal(history, alpha=1.5):
    """이미 mastered한 source를 sample (catastrophic forgetting 방지)."""
    mastered = history.mastered_offline()
    return random.choice(mastered) if mastered else random_choice(all)
```

내 추천: 둘 다 별개 옵션으로 두기:
- `cfg.exploration_p`: 안 배운 것 우선
- `cfg.rehearsal_p`: 마스터한 것 다시
- 두 비율은 독립적으로 plan_maker에서 합쳐 적용

### H. **Source 단위 통계 입자 통일** (low risk, eval correctness)

`DatasetHistory.record`의 source_id 정의 통일:
```python
# tsg도 cfg별로 분리
source_id = f"tsg:slice={cfg.init_slice[0]}x{cfg.init_slice[1]}x{cfg.init_slice[2]}"
```

→ 각 cfg variant 별 학습 진행도 측정 가능.

### I. **Bootstrap Phase 정책 명시화** (low risk, debuggability)

현재: `OVERFIT_TARGET_PATH` 1개 + preview_cnt 강제 + rollout_idx=0. 이유 주석 거의 없음.

```python
# AgentConfig에 명시
@dataclass
class BootstrapPolicy:
    enabled: bool = True
    fixed_source: Optional[str] = None       # None이면 정상 풀
    fixed_preview_cnt: Optional[int] = None
    fixed_rollout_seed: bool = False
    duration_ratio: float = 0.10

cfg.bootstrap = BootstrapPolicy(
    fixed_source=OVERFIT_TARGET_PATH,
    fixed_preview_cnt=PREVIEW_MAX,
    fixed_rollout_seed=True,
)
```

bootstrap에서 무엇을 강제하는지 한눈에 보임. config 끄기/켜기 쉬움.

### J. **`OVERFIT_TARGET_PATH` 모듈 상수 → Config 필드** (very low risk)

```python
# AgentConfig
overfit_target_path: str = "planning/data/Item_data/paper/setting123_discrete/dataset_episode_012.json"
```

상수 박혀 있어 Config 흐름과 분리되어 있던 것 정리.

---

## 4. 권장 적용 순서

| 순서 | 항목 | 영향 | 위험 | 상태 |
|---|---|---|---|---|
| ~~1~~ | ~~**C. bin과 dataset 분리**~~ | 의미 정리 + 실험 다양성 ↑ | 매우 낮음 | ✅ 적용 (B 추상화에 흡수, Sec 2-3) |
| 2 | **D. TSG cfg 자동 동기화** | 미스매치 버그 방지 | 매우 낮음 | ⏸ 보류 (Sec 2-6 부분 해결, bin_size↔BIN_SPECS sync는 미해결) |
| ~~3~~ | ~~**H. source_id 입자 통일** (synthetic도 cfg별)~~ | history 정확도 ↑ | 매우 낮음 | ✅ 적용 (B 추상화에 흡수, Sec 2-7) |
| ~~4~~ | ~~**G. rehearsal 정리** (의미 명확화)~~ | 코드 가독성 + mastered_recorded 활용 | 낮음 | ✅ 적용 (선택 B, Sec 2-4) |
| ~~5~~ | ~~**I. Bootstrap policy 명시화**~~ | 디버깅 편의성 | 낮음 | ✅ 적용 (BootstrapPolicy, Sec 2-5) |
| ~~6~~ | ~~**J. OVERFIT_TARGET_PATH → Config**~~ | 정리 | 매우 낮음 | ✅ 적용 (BootstrapPolicy.fixed_recorded_path default, Sec 2-5/2-12) |
| ~~7~~ | ~~**A. 모드 명명 재정의**~~ | 가독성 ↑ | 중간 | ✅ 적용 (옵션 C) |
| 8 | **E. Difficulty-weighted curriculum** | 학습 효율 ↑ | 중간 | ⏸ 보류 (B 적용 후 history 활용 가능해짐) |
| ~~9~~ | ~~**F. eval/train plan_maker 통합**~~ | 평가 신뢰성 ↑ + 한 곳에서 분포 명시 | 중간 | ✅ 적용 (EvalPolicy + `_make_plans_core` 재사용, Sec 2-8) |
| ~~10~~ | ~~**B. DataSource 추상화**~~ | 확장성 ↑ | 높음 | ✅ 적용 (Sec 2-2/2-3/2-7/2-10 통합 해결) |

→ **2026-05-04 시점 — A, B, C, H 적용 완료.** Sec 2-2/2-3/2-7/2-10이 B 한 번으로 통합 해결.
→ 남은 항목(D bin sync / G rehearsal / I bootstrap / J OVERFIT path / E curriculum / F eval/train 통합)은 학습 결과 보고 결정.

---

## 5. 요약

```
=== 현재 ===
3 mode (offline / online_type / tsg) 분기가 4곳 (env, plan_maker, packer, history)에 흩어짐.
모드 이름이 RL 문헌의 일반 의미와 충돌.
bin alias가 data mode에 의존 → 강한 coupling.
rehearsal 이름과 동작 어긋남.
DatasetHistory는 통계만 쌓고 학습 영향 0.
eval/train plan_maker 분리되어 분포 다름.

=== 개선 핵심 ===
1) bin과 dataset 분리 (orthogonal하게)
2) TSG cfg와 bin spec 자동 동기화
3) source_id 입자 통일 (tsg도 cfg별 분리)
4) rehearsal 정의 명확화 (탐색 vs 진짜 복습 분리)
5) (장기) DataSource Protocol로 모드 분기 4곳 → 1곳 통합
```
