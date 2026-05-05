"""
Recursive Guillotine Partitioning 기반 synthetic 데이터셋 생성기.

bin 전체(=SU 100% 가능 영역)를 단방향 split만으로 잘라 박스 시퀀스를 만든다.
- 모든 cut에서 min_item_mm + max_aspect_ratio 제약 강제 → 극단적 박스 방지
- volume-weighted leaf 선택 → 큰 leaf 우선 split → 결과 분포가 고른 중간 크기
- bottom-up (z, y, x) sort → 학습 시 안정적 적재 순서
- margin은 후처리 차감 (bin 좌하단 정렬 보존, 박스 사이에만 gap)

옛 `TrainsetGenerator`(init_slice + merge/split 양방향)는 제어가 어렵고 극단적
조각이 자주 생겼음 → 2026-05-04 완전 교체.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SyntheticConfig:
    """Recursive guillotine 생성기 설정."""
    bin_size: Tuple[int, int, int] = (1000, 1000, 1000)   # (W, H, D) in mm
    max_items: int = 30                                    # 박스 개수 상한 (직접 제어)
    min_item_mm: int = 100                                 # 모든 박스 min(w,h,d) ≥ 이 값
    max_aspect_ratio: float = 3.0                          # max(w,h,d) / min(w,h,d) ≤ 이 값
    margin_x: int = 0                                      # 후처리 차감 (박스 사이 gap)
    margin_y: int = 0
    seed: Optional[int] = None
    # tuning
    cut_retry: int = 20                                    # aspect ratio fail 시 재시도 횟수


# ─────────────────────────────────────────────────────────────────────
# Internal box
# ─────────────────────────────────────────────────────────────────────
class _Box:
    __slots__ = ("x", "y", "z", "width", "height", "depth")

    def __init__(self, x, y, z, w, h, d):
        self.x = int(x); self.y = int(y); self.z = int(z)
        self.width = int(w); self.height = int(h); self.depth = int(d)

    @property
    def volume(self) -> int:
        return self.width * self.height * self.depth

    def size_along(self, axis: str) -> int:
        if axis == "x": return self.width
        if axis == "y": return self.height
        if axis == "z": return self.depth
        raise ValueError(axis)

    def other_two(self, axis: str) -> Tuple[int, int]:
        if axis == "x": return (self.height, self.depth)
        if axis == "y": return (self.width,  self.depth)
        if axis == "z": return (self.width,  self.height)
        raise ValueError(axis)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _check_aspect(dims: Tuple[int, ...], max_ratio: float) -> bool:
    """모든 dim 0 초과, max/min ≤ max_ratio."""
    if any(d <= 0 for d in dims):
        return False
    return max(dims) / min(dims) <= max_ratio + 1e-9


def _pick_valid_cut(
    box: _Box, axis: str, cfg: SyntheticConfig, rng: random.Random
) -> Optional[int]:
    """주어진 axis에서 min_size + aspect ratio 제약을 만족하는 cut 위치 1개를 sample.
    실패 시 None."""
    size = box.size_along(axis)
    other = box.other_two(axis)
    lo = cfg.min_item_mm
    hi = size - cfg.min_item_mm
    if hi < lo:
        return None

    # uniform random + aspect ratio retry
    for _ in range(cfg.cut_retry):
        c = rng.randint(lo, hi)
        # 두 조각 모두 aspect ratio 통과해야 함
        if (_check_aspect((c,) + other, cfg.max_aspect_ratio)
                and _check_aspect((size - c,) + other, cfg.max_aspect_ratio)):
            return c
    return None


def _split_at(box: _Box, axis: str, cut: int) -> List[_Box]:
    if axis == "x":
        return [
            _Box(box.x,         box.y, box.z, cut,              box.height, box.depth),
            _Box(box.x + cut,   box.y, box.z, box.width - cut,  box.height, box.depth),
        ]
    if axis == "y":
        return [
            _Box(box.x, box.y,         box.z, box.width, cut,               box.depth),
            _Box(box.x, box.y + cut,   box.z, box.width, box.height - cut,  box.depth),
        ]
    # z
    return [
        _Box(box.x, box.y, box.z,        box.width, box.height, cut),
        _Box(box.x, box.y, box.z + cut,  box.width, box.height, box.depth - cut),
    ]


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────
def generate_synthetic_items(cfg: SyntheticConfig) -> List[Dict]:
    """Recursive guillotine partitioning + 후처리 margin.

    Returns:
        list of dict — packer/Item 호환 포맷:
            {partno, name, objshape, width, height, depth, rotation_quat,
             priority, updown, options, weight, loadbear, unit, b_position}
    """
    rng = random.Random(cfg.seed) if cfg.seed is not None else random.Random()

    leaves: List[_Box] = [_Box(0, 0, 0, *cfg.bin_size)]
    failed: set = set()   # split 시도해서 실패한 leaf의 id (재시도 안 함)

    while len(leaves) < cfg.max_items:
        # split 가능 후보만
        candidates = [b for b in leaves if id(b) not in failed]
        if not candidates:
            break

        # volume-weighted leaf 선택 (큰 leaf 우선 → 결과 분포가 고름)
        weights = [b.volume for b in candidates]
        target = rng.choices(candidates, weights=weights, k=1)[0]

        # axis 무작위 순서로 시도 → 첫 번째 valid cut 채택
        axes = ["x", "y", "z"]
        rng.shuffle(axes)
        new_boxes: Optional[List[_Box]] = None
        for axis in axes:
            cut = _pick_valid_cut(target, axis, cfg, rng)
            if cut is not None:
                new_boxes = _split_at(target, axis, cut)
                break

        if new_boxes is None:
            # 어떤 axis로도 split 불가 → unsplittable 표시 (재시도 안 함)
            failed.add(id(target))
            continue

        leaves.remove(target)
        leaves.extend(new_boxes)

    # bottom-up 순서 (z, y, x)
    leaves.sort(key=lambda b: (b.z, b.y, b.x))

    # margin 후처리: width/height에서 차감.
    # bin 좌하단(x=0, y=0) 정렬은 보존되고 박스 사이/우상단에만 gap → bin 가장자리 활용 ↑
    if cfg.margin_x > 0 or cfg.margin_y > 0:
        for b in leaves:
            if cfg.margin_x > 0:
                b.width = max(1, b.width - cfg.margin_x)
            if cfg.margin_y > 0:
                b.height = max(1, b.height - cfg.margin_y)

    return [
        {
            "partno": str(i),
            "name": f"synth_{i}",
            "objshape": "cube",
            "width": b.width,
            "height": b.height,
            "depth": b.depth,
            "rotation_quat": [0, 0, 0, 1],
            "priority": 7,
            "updown": False,
            "options": {"color": "#14ba5e"},
            "weight": 0,
            "loadbear": 0,
            "unit": "mm",
            "b_position": [b.x, b.y, b.z],
        }
        for i, b in enumerate(leaves)
    ]


# ─────────────────────────────────────────────────────────────────────
# 디버그 진입점: 생성된 박스를 bin.render()로 시각화
#   사용 예) python -m planning.data.synthetic_generator --seed 0 --max-items 30 --margin-x 0
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from planning.bin import Bin
    from planning.item import Item, RotationType

    parser = argparse.ArgumentParser(
        description="synthetic_generator 결과를 bin.render()로 시각화"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--min-item-mm", type=int, default=100)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--bin-w", type=int, default=1000)
    parser.add_argument("--bin-h", type=int, default=1000)
    parser.add_argument("--bin-d", type=int, default=1000)
    parser.add_argument("--margin-x", type=int, default=0)
    parser.add_argument("--margin-y", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=str, default="planning/data/_synthetic_preview",
        help="render 저장 폴더"
    )
    parser.add_argument("--show", action="store_true", help="GUI 창으로 띄우기")
    parser.add_argument(
        "--n-grids", type=int, default=1,
        help="여러 seed로 grid 형태로 출력할 개수 (1이면 단일 ep)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _render_one(seed: int) -> tuple[Bin, list[dict], int]:
        """1 episode 박스 생성 → bin에 그대로 store → render."""
        cfg = SyntheticConfig(
            bin_size=(args.bin_w, args.bin_h, args.bin_d),
            max_items=args.max_items,
            min_item_mm=args.min_item_mm,
            max_aspect_ratio=args.max_aspect_ratio,
            margin_x=args.margin_x,
            margin_y=args.margin_y,
            seed=seed,
        )
        item_dicts = generate_synthetic_items(cfg)

        # bin 생성
        b = Bin(
            partno="synth_preview",
            name=f"synth_seed{seed}",
            width=args.bin_w,
            height=args.bin_h,
            depth=args.bin_d,
            unit="mm",
        )

        # 박스를 b_position 그대로 store (이미 cut 결과라 위치 확정)
        for d in item_dicts:
            it = Item(**d)
            b.store(it)

        # 부피 합산 통계
        total_v = sum(it["width"] * it["height"] * it["depth"] for it in item_dicts)
        bin_v = args.bin_w * args.bin_h * args.bin_d
        ratio = total_v / bin_v
        # bin.SU는 실제 dead volume 계산을 거치므로 margin이 있으면 1.0 미만일 수 있음
        return b, item_dicts, total_v, ratio

    print(f"[synth-render] {args.n_grids} ep × seeds [{args.seed}..{args.seed + args.n_grids - 1}]")
    print(f"               bin=({args.bin_w}×{args.bin_h}×{args.bin_d}) max_items={args.max_items} "
          f"min_dim={args.min_item_mm} aspect≤{args.max_aspect_ratio} "
          f"margin=({args.margin_x},{args.margin_y})")
    print()

    for i in range(args.n_grids):
        seed = args.seed + i
        b, items, total_v, ratio = _render_one(seed)

        # 통계
        dims = [d for it in items for d in (it["width"], it["height"], it["depth"])]
        aspects = [
            max(it["width"], it["height"], it["depth"])
            / max(1e-6, min(it["width"], it["height"], it["depth"]))
            for it in items
        ]
        print(f"[seed={seed}] n={len(items):3d}  "
              f"vol_ratio={ratio:.4f}  bin.SU={float(b.SU):.4f}  "
              f"dim min={min(dims):3d} max={max(dims):3d} mean={sum(dims)/len(dims):.0f}  "
              f"aspect max={max(aspects):.2f}")

        # render
        name = f"seed{seed:03d}_n{len(items)}_min{args.min_item_mm}_aspect{args.max_aspect_ratio}_margin{args.margin_x}-{args.margin_y}"
        try:
            b.render(
                save=True,
                save_path=str(out_dir),
                name=name,
                size_annotation=False,
                show=args.show,
                write_num=True,
            )
            print(f"           → saved: {out_dir / ('result_' + b.name + '_' + name + '.jpg')}")
        except Exception as e:
            print(f"           ⚠️ render failed: {e}")
            import traceback; traceback.print_exc()
