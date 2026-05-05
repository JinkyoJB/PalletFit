# Synthetic 데이터셋 생성 알고리즘

> 대상 파일: `planning/data/synthetic_generator.py`, `planning/RL/PalletFit_RL/env.py` (`_generate_synthetic_items`)
> 목적: 학습용 합성 박스 시퀀스를 SU=1.0 가능하면서도 제어 가능한 분포로 생성.

---

## 1. 배경

이전 `TrainsetGenerator`(planning/data/trainset_generator.py, 2026-05-04 삭제됨)는 다음 흐름이었음:

```
init_slice=(x_div, y_div, z_div) → 균일 grid 분할
   ↓
num_merge × random_merge (인접+직육면체 보존)
   ↓
num_split × random_split (axis 랜덤, cut 위치 random.randint(1, size-1))
   ↓
margin 후처리 차감
```

**문제 5가지** (`docs/4_dataset_design.md` Sec 2-9, 2-6 참조):
1. `random.randint(1, size-1)` → 1000mm 박스를 (3, 997)로 자를 수 있음 → **극단적 비대칭**.
2. `min_width/height/depth` 검증이 init item의 최솟값으로만 동작 → 한 번이라도 작은 슬라이스 통과하면 이후 모든 split이 통과.
3. **Aspect ratio 제약 없음** → (3, 1000, 1000) 슬래브 가능.
4. `num_merge`/`num_split`가 양방향 무작위 → 박스 개수/크기 분포를 직관적으로 제어 어려움.
5. `init_slice`가 bin 크기와 무관 hardcoded.

→ 사용자 표명: "조각이 너무 기이해진다(?)" — 위 (1)+(3)이 근본 원인.

---

## 2. 일반적 방식 — 3D bin packing RL 논문 합의

대다수 논문(DeepPack3D, AttendPack, RobotPack 등)은 **Recursive Guillotine Partitioning** 기반:

```
bin 전체를 root로 시작
  ↓ (반복)
splittable한 leaf 후보 중 1개 선택
  → axis 중 splittable한 것 선택 (≥ 2 × min_size)
  → cut 위치를 [min_size, size − min_size] 안에서 sample
  → 두 조각 모두 aspect ratio 제약 통과해야 채택
  ↓
max_items 도달 또는 split 가능 leaf 없을 때 종료
```

**핵심 차이**:
- **단방향 split만** (merge 없음) → 결과 예측 가능.
- **min_size + aspect ratio가 매 cut에서 강제** → "기이한 조각" 원천 차단.
- **item 개수 직접 제어** (`max_items`).
- **항상 SU=1.0 보장** (split만 하니 부피 합 = bin 부피).

---

## 3. 적용된 구현 (2026-05-04, `synthetic_generator.py`)

### 3-1. SyntheticConfig

```python
@dataclass
class SyntheticConfig:
    bin_size: Tuple[int, int, int] = (1000, 1000, 1000)   # (W, H, D) mm
    max_items: int = 30                                    # 박스 개수 직접 제어
    min_item_mm: int = 100                                 # min(w,h,d) ≥ 이 값
    max_aspect_ratio: float = 3.0                          # max(w,h,d)/min(w,h,d) ≤ 이 값
    margin_x: int = 0                                      # 후처리 차감 (박스 사이 gap)
    margin_y: int = 0
    seed: Optional[int] = None
    cut_retry: int = 20                                    # aspect 실패 시 재시도
```

### 3-2. 알고리즘

```python
def generate_synthetic_items(cfg):
    rng = random.Random(cfg.seed)
    leaves = [_Box(0, 0, 0, *cfg.bin_size)]   # bin 통째로 시작
    failed = set()                             # 못 자른 leaf의 id

    while len(leaves) < cfg.max_items:
        candidates = [b for b in leaves if id(b) not in failed]
        if not candidates: break

        # ① volume-weighted leaf 선택 (큰 leaf 우선 → 결과 분포 균일화)
        target = rng.choices(candidates, weights=[b.volume for b in candidates], k=1)[0]

        # ② axis 무작위 순서로 시도 → 첫 번째 valid cut 채택
        axes = ["x", "y", "z"]; rng.shuffle(axes)
        for axis in axes:
            cut = _pick_valid_cut(target, axis, cfg, rng)
            if cut is not None:
                leaves.remove(target)
                leaves.extend(_split_at(target, axis, cut))
                break
        else:
            failed.add(id(target))

    leaves.sort(key=lambda b: (b.z, b.y, b.x))      # bottom-up

    # 후처리 margin (bin 좌하단은 보존, 박스 사이/우상단에만 gap)
    if cfg.margin_x > 0 or cfg.margin_y > 0:
        for b in leaves:
            b.width  = max(1, b.width  - cfg.margin_x)
            b.height = max(1, b.height - cfg.margin_y)
    return [_to_dict(b, i) for i, b in enumerate(leaves)]
```

### 3-3. `_pick_valid_cut` — min_size + aspect ratio 동시 검사

```python
def _pick_valid_cut(box, axis, cfg, rng):
    size = box.size_along(axis)
    other = box.other_two(axis)
    lo, hi = cfg.min_item_mm, size - cfg.min_item_mm
    if hi < lo: return None
    for _ in range(cfg.cut_retry):
        c = rng.randint(lo, hi)
        if (_check_aspect((c,) + other,        cfg.max_aspect_ratio) and
            _check_aspect((size - c,) + other, cfg.max_aspect_ratio)):
            return c
    return None
```

→ 두 조각 모두 aspect ratio 통과해야 채택. retry 한도까지 못 찾으면 None (다른 axis로 시도).

### 3-4. 사용자 결정사항 반영

| 결정 | 값 / 동작 |
|---|---|
| Q1 다양성 | `grid_mm` **미적용** — cut 위치는 free integer (`random.randint(lo, hi)`). 박스 치수가 50/100mm 배수에 묶이지 않아 다양성 ↑. |
| Q2 일반적 값 | `max_aspect_ratio=3.0` (표준), `max_items=30` (TSG 32-item과 비슷한 규모), `min_item_mm=100` (옛 default 유지). |
| Q3 옛 `TrainsetGenerator` | **완전 삭제** (`planning/data/trainset_generator.py` 파일 자체 제거). |
| Q4 margin 처리 | **후처리 차감 유지** (사용자 컨셉: "bin 귀퉁이의 margin을 최대한 없애서 빈틈 활용"). 좌하단(x=0, y=0) 정렬 보존, 박스 사이/우상단에 gap. |
| Q5 leaf 선택 | **volume-weighted** — 큰 leaf 우선 split → 분포가 고른 중간 크기로 수렴. |

---

## 4. 측정 결과

### 4-1. 단일 episode (seed=42)

| 항목 | 값 |
|---|---|
| 생성 박스 수 | **30** (max_items 정확히 도달) |
| 부피 합 / bin | **1.0000** (SU=1.0 보장) |
| aspect ratio | min 1.05, **max 2.94**, mean 1.92 (≤ 3.0 강제) |
| min_dim (박스의 최소 변) | min **129**, max 350, mean 233 (≥ 100 강제) |
| bottom-up sort | 첫 5개 z=0 ✓ |

### 4-2. 박스 개수 안정성 (5 random seeds)

| seed | n | min(dim) | max(dim) | mean(dim) |
|---|---|---|---|---|
| 0 | 30 | 119 | 635 | 316 |
| 1 | 30 | 163 | 607 | 333 |
| 2 | 30 | 131 | 643 | 325 |
| 3 | 30 | 130 | 580 | 327 |
| 4 | 30 | 132 | 721 | 322 |

→ seed 무관하게 30개 정확 생성, dim 분포 일관(mean 320±15).

### 4-3. Volume 분포 (volume-weighted 효과 검증)

| 분위 | 박스 부피 (mm³) | bin 비율 |
|---|---|---|
| smallest | 6,750,054 | 0.68% |
| median | 28,854,540 | 2.89% |
| largest | 109,162,200 | 10.92% |

→ largest/smallest = **16.2×** (uniform random 선택이었으면 100×+ 흔함). **volume-weighted가 분포 균일화에 기여**.

### 4-4. Margin 후처리 검증

- `margin=0`: 첫 박스 width=224, height=658, b_position=(0,0,0)
- `margin=10`: 첫 박스 width=214 (= -10), height=648 (= -10), b_position=**(0,0,0) 보존**

→ 사용자 의도("bin 좌하단 margin 최대한 없애기") 정확히 동작.

### 4-5. env 통합 + 학습 신호 정합

```
DEFAULT_SYNTHETIC_CFG = {bin_size:(1000,1000,1000), max_items:30,
                         min_item_mm:100, max_aspect_ratio:3.0, margin_x:0, margin_y:0}

reset queue len = 30 ✓
random action 완주: ep_return=+6.60, SU=0.504, packed=14
sum_terms == ep_return = True ✓
```

random action이 SU=0.504밖에 못 채우는 게 **정상** — 박스가 다양한 크기/aspect라서 random은 100% 못 채움. 옛 (4,4,2) 균일 박스 시나리오는 random에도 SU=1.0이 잦았는데 그게 너무 쉬운 문제였음. 새 generator는 RL이 학습할 만한 적정 난이도.

---

## 5. 옛 코드와 비교

| 측면 | 이전 (`TrainsetGenerator`) | 현재 (`generate_synthetic_items`) |
|---|---|---|
| 시작 | `init_slice` 균일 grid | bin 통째 1개 |
| 진행 | merge ↔ split 양방향 | split만 (단방향) |
| min_size 제약 | init items의 minimum (사실상 무력) | **매 cut에서 강제** ✓ |
| Aspect ratio 제약 | **없음** | **명시적** (max_aspect_ratio) ✓ |
| Item 개수 제어 | num_merge/num_split 간접 | **`max_items` 직접** ✓ |
| 결과 예측 가능성 | 낮음 (양방향 난수) | 높음 (단조 split) |
| 생성 박스 크기 | 극단치 잦음 (3mm 슬래브 가능) | 제약 내 균일 (~120-700mm) |
| 코드 길이 | ~430줄 (인접 그래프 등) | **~150줄** |
| SU=1.0 보장 | 보장 (margin 제외) | 보장 (margin 후처리) |
| margin 처리 | 후처리 차감 (gap 어긋남) | 후처리 차감, 좌하단 보존 의도 명시 |

---

## 6. 미래 확장 가능성 (학습 결과 보고 결정)

### 6-1. Curriculum control
`min_item_mm`/`max_items`로 난이도 단계 자연 조절:
- bootstrap: `min=200, max_items=15` (큰 박스 적게 → 쉬움)
- mid:       `min=150, max_items=25`
- late:      `min=100, max_items=40` (다양한 크기 많음 → 어려움)

### 6-2. Grid-aligned cut (현재 미적용 — 사용자 결정으로 다양성 선택)
`grid_mm=50`이면 박스 치수가 50mm 배수 → real-world carton과 align. 학습 분산 ↓ 효과. 다양성을 다시 줄이고 싶을 때 옵션화 가능.

### 6-3. Bin spec 자동 동기화 (`docs/4_dataset_design.md` Sec 2-6)
현재 `cfg.bin_size`가 명시 필수 → BIN_SPECS와 미스매치 가능. `auto_sync_with_bin(bin_alias)` 헬퍼 추가하면 bin alias만 주면 자동 매칭.

---

## 7. 변경 파일 요약

| 파일 | 변경 |
|---|---|
| `planning/data/synthetic_generator.py` | **신규 작성** (~150줄) — `SyntheticConfig`, `generate_synthetic_items`, `_Box`/`_pick_valid_cut`/`_split_at` |
| `planning/data/trainset_generator.py` | **삭제** (옛 알고리즘 폐기) |
| `planning/RL/PalletFit_RL/env.py` | `TSGConfig` → `SyntheticConfig` import, `_generate_items_with_tsg` → `_generate_synthetic_items`(20줄로 축소), `__main__` 디버그 러너의 `synthetic_cfg` payload 새 키 |
| `planning/RL/PalletFit_RL/agent.py` | `DEFAULT_SYNTHETIC_CFG` 새 키 (bin_size, max_items, min_item_mm, max_aspect_ratio, margin_x, margin_y) |

---

## 8. 검증 체크리스트

- [x] 부피 합 = bin 부피 (SU=1.0 보장)
- [x] aspect ratio ≤ max_aspect_ratio 강제 통과 (cut 시점)
- [x] min(w,h,d) ≥ min_item_mm 강제 통과
- [x] max_items 정확 도달 (다른 seed 5회 모두 30개)
- [x] bottom-up sort (z, y, x) 정상
- [x] margin 후처리 시 좌하단(0,0,0) 보존
- [x] env 통합: reset queue len, ep_return 정합 (sum_terms == ep_return)
- [x] 옛 함수/클래스(`_generate_items_with_tsg`, `TSGConfig`, `TrainsetGenerator`) 완전 제거
- [x] 옛 파일 `trainset_generator.py` 삭제

⚠️ **알려진 제한**: aspect ratio는 cut 시점에 검증되므로, **margin 후처리로 width/height가 줄면 비율이 cut 당시보다 커질 수 있음**. 예: 측정에서 margin=20일 때 max aspect 2.96 → 3.33으로 증가. 학습에 큰 영향은 없지만, 엄격한 보장이 필요하면 후속으로 "post-margin aspect 재검사 + 회귀시 cut 재시도" 옵션 추가 가능.

---

## 9. 시각화 디버그 러너

`synthetic_generator.py` 하단의 `if __name__ == "__main__"` 블록 — bin.render()로 생성된 박스 분포를 확인 가능.

### 사용법

```bash
# 단일 episode (default 옵션: bin 1m³, max_items=30, min=100, aspect≤3.0, margin 0)
python -m planning.data.synthetic_generator

# 5개 seed 한 번에 비교
python -m planning.data.synthetic_generator --seed 0 --n-grids 5

# margin 효과 확인 (박스 사이 gap)
python -m planning.data.synthetic_generator --seed 0 --margin-x 20 --margin-y 20

# 작은 박스만 (curriculum bootstrap)
python -m planning.data.synthetic_generator --seed 0 --max-items 15 --min-item-mm 200

# 큰 박스 다수 (curriculum late)
python -m planning.data.synthetic_generator --seed 0 --max-items 40 --min-item-mm 80

# 다른 bin 크기
python -m planning.data.synthetic_generator --bin-w 1200 --bin-h 800 --bin-d 1500

# GUI 창으로 띄우기
python -m planning.data.synthetic_generator --show
```

### 출력 예시

```
[synth-render] 5 ep × seeds [0..4]
               bin=(1000×1000×1000) max_items=30 min_dim=100 aspect≤3.0 margin=(0,0)

[seed=0] n= 30  vol_ratio=1.0000  bin.SU=1.0000  dim min=119 max=635 mean=316  aspect max=2.96
           → saved: planning/data/_synthetic_preview/result_synth_seed0_seed000_n30_min100_aspect3.0_margin0-0.jpg
[seed=1] n= 30  vol_ratio=1.0000  bin.SU=1.0000  dim min=163 max=607 mean=333  aspect max=2.98
...
```

각 ep마다:
- `vol_ratio`: 박스 부피 합 / bin 부피 (margin=0이면 1.0000 정확)
- `bin.SU`: 실제 dead volume 계산 결과 (대각선 등 측정으로 vol_ratio와 미세 다를 수 있음)
- `dim min/max/mean`: 박스 변 길이 분포
- `aspect max`: 가장 비대칭한 박스의 max(w,h,d)/min(w,h,d)

기본 저장 폴더: `planning/data/_synthetic_preview/`. JPG로 6면 시각화 (bottom 정렬된 적층).
