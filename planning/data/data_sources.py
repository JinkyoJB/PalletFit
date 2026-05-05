"""
DataSource 추상화 — env/plan_maker/history가 mode별 분기 없이 단일 인터페이스만 사용.

이전: env._build_items_from_plan, env._build_binPacker_from_plan, agent._make_plans_core,
       agent.DatasetHistory.record 4곳에서 `if mode == "recorded" / elif "type_sampled" / elif "synthetic"`
       분기 중복. 새 mode 추가 시 4-5곳 수정 필요.

현재: 모든 박스 source가 DataSource 인터페이스 구현 → env는 `make_source(spec).sample(seed)` 한 줄.
       새 mode = 클래스 1개 추가 + _SOURCE_REGISTRY에 등록 = 2곳만 수정.

Plan에 들어가는 spec dict 형식:
    {"mode": "recorded",     "args": {"paths": [...]}}
    {"mode": "type_sampled", "args": {"paths": [...], "n_items": 30}}
    {"mode": "synthetic",    "args": {bin_size, max_items, min_item_mm, max_aspect_ratio, ...}}

spec dict는 SubprocVecEnv IPC pickle에 안전 (객체 아님).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional
import json
import random

from utils.util_functions import load_offline_data
from planning.data.synthetic_generator import SyntheticConfig, generate_synthetic_items


# ─────────────────────────────────────────────────────────────────────
# 공통 helper
# ─────────────────────────────────────────────────────────────────────
def _load_type_templates(path: str) -> List[Dict]:
    """파일에서 type 거푸집 dict 리스트 로드."""
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        if path.endswith(".txt"):
            return f.read().splitlines()
        if path.endswith(".csv"):
            return f.read().split(",")
    raise ValueError(f"Unsupported file extension: {path}")


def _sample_from_templates(
    templates: List[Dict],
    n_items: int,
    rng: random.Random,
    weights: Optional[List[float]] = None,
) -> List[Dict]:
    """Type 거푸집에서 n_items개 box dict sample.

    docs/4 Sec 2-11 해결 (2026-05-05): 옛 동작은 매 호출마다 weights를 random 재배정해서
    같은 파일이라도 episode마다 분포가 임의로 바뀌었음 → 의미 불명.
    이제는 weights=None이면 uniform(모든 type 동일 확률), 사용자 명시하면 그 분포 사용.

    Args:
        templates: 박스 type 거푸집 dict 리스트
        n_items: sample할 박스 수
        rng: random.Random 인스턴스 (seed 영향)
        weights: type별 sampling 확률(상대값). None이면 uniform. 길이는 templates와 일치해야 함.
    """
    if not templates:
        return []
    if weights is None:
        weights = [1.0] * len(templates)
    elif len(weights) != len(templates):
        raise ValueError(
            f"weights length {len(weights)} != templates length {len(templates)}"
        )
    sampled: List[Dict] = []
    for i in range(int(n_items)):
        idx = rng.choices(range(len(templates)), weights=weights)[0]
        d = dict(templates[idx])         # shallow copy
        d["partno"] = str(i)             # ID 새로
        sampled.append(d)
    return sampled


# ─────────────────────────────────────────────────────────────────────
# DataSource ABC
# ─────────────────────────────────────────────────────────────────────
class DataSource(ABC):
    """박스 시퀀스를 sample하는 추상 인터페이스."""

    mode: str = ""   # mode 식별자 (history bucket key, log tag 등)

    @abstractmethod
    def sample(self, seed: int) -> List[Dict]:
        """seed 기반으로 박스 dict 리스트 반환 (Item 호환 포맷)."""
        ...

    def source_id(self) -> str:
        """DatasetHistory 키 — 같은 source 내에서 유의미한 분리(파일 경로/cfg variant 등)."""
        return self.mode

    def bin_alias_hint(self) -> Optional[str]:
        """이 source가 선호하는 bin alias (env가 plan["bin"] 없을 때 fallback). None이면 안 강제."""
        return None


# ─────────────────────────────────────────────────────────────────────
# Concrete sources
# ─────────────────────────────────────────────────────────────────────
class RecordedSource(DataSource):
    """미리 정해진 박스 시퀀스 파일에서 로드 (옛 'offline')."""
    mode = "recorded"

    def __init__(self, paths: List[str], **_unused):
        self.paths = list(paths)
        self._last_chosen: Optional[str] = None

    def sample(self, seed: int) -> List[Dict]:
        if not self.paths:
            return []
        rng = random.Random(seed)
        self._last_chosen = rng.choice(self.paths)
        items = load_offline_data(self._last_chosen) or []
        return list(items)

    def source_id(self) -> str:
        return f"recorded:{self._last_chosen}" if self._last_chosen else "recorded:unknown"

    def bin_alias_hint(self) -> Optional[str]:
        return "experiment_RL"


class TypeSampledSource(DataSource):
    """타입 거푸집 정의에서 sampling (옛 'online_type').

    docs/4 Sec 2-11 해결: weights를 명시할 수 있게 함. None이면 uniform.
    옛 매 episode random 재배정 동작은 제거 (의미 불명이었음).
    """
    mode = "type_sampled"

    def __init__(
        self,
        paths: List[str],
        n_items: int = 30,
        weights: Optional[List[float]] = None,
        **_unused,
    ):
        self.paths = list(paths)
        self.n_items = int(n_items)
        # weights: None이면 uniform, list면 type 거푸집별 sampling 확률
        self.weights = list(weights) if weights is not None else None
        self._last_chosen: Optional[str] = None

    def sample(self, seed: int) -> List[Dict]:
        if not self.paths:
            return []
        rng = random.Random(seed)
        self._last_chosen = rng.choice(self.paths)
        templates = _load_type_templates(self._last_chosen)
        return _sample_from_templates(templates, self.n_items, rng, weights=self.weights)

    def source_id(self) -> str:
        return f"type_sampled:{self._last_chosen}" if self._last_chosen else "type_sampled:unknown"

    def bin_alias_hint(self) -> Optional[str]:
        return "default2"


class SyntheticSource(DataSource):
    """Recursive guillotine partitioning (옛 'tsg', synthetic_generator.py 활용).

    docs/4 Sec 2-6 해결 (2026-05-05): bin_size를 BIN_SPECS와 자동 동기화.
    우선순위: 명시 cfg_kwargs['bin_size'] > bin_alias의 BIN_SPECS lookup > 기본값 (1m³).

    Args:
        bin_alias: BIN_SPECS 키. 명시되면 그 bin의 (W, H, D)를 bin_size로 자동 사용.
                   cfg_kwargs에 bin_size가 명시되면 그게 우선 (override).
                   BIN_SPECS에 없는 alias는 silent skip (env에서 별도 처리).
        **cfg_kwargs: SyntheticConfig 필드와 동일.
    """
    mode = "synthetic"

    def __init__(self, bin_alias: Optional[str] = None, **cfg_kwargs):
        cfg_kwargs = dict(cfg_kwargs)
        # bin_alias가 명시되고 bin_size가 명시되지 않은 경우만 자동 주입
        if bin_alias and "bin_size" not in cfg_kwargs:
            from planning.BinSpecsDict import BIN_SPECS
            spec = BIN_SPECS.get(bin_alias)
            if spec is not None:
                cfg_kwargs["bin_size"] = (
                    int(spec["width"]), int(spec["height"]), int(spec["depth"]),
                )
        self._bin_alias = bin_alias
        self._cfg_kwargs = cfg_kwargs

    def sample(self, seed: int) -> List[Dict]:
        cfg = SyntheticConfig(**self._cfg_kwargs)
        cfg = replace(cfg, seed=int(seed))
        return generate_synthetic_items(cfg)

    def source_id(self) -> str:
        c = self._cfg_kwargs
        return (f"synthetic:n={c.get('max_items', 30)}"
                f"_min={c.get('min_item_mm', 100)}"
                f"_asp={c.get('max_aspect_ratio', 3.0)}")

    def bin_alias_hint(self) -> Optional[str]:
        # SyntheticSource가 bin_alias를 알면 그걸 hint로 쓸 수 있음 → bin/data 일관성
        return self._bin_alias


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────
_SOURCE_REGISTRY: Dict[str, type] = {
    "recorded":     RecordedSource,
    "type_sampled": TypeSampledSource,
    "synthetic":    SyntheticSource,
}


def register_source(klass: type) -> None:
    """새 DataSource 클래스 등록 (외부 확장용)."""
    if not issubclass(klass, DataSource):
        raise TypeError(f"{klass} is not a DataSource subclass")
    if not getattr(klass, "mode", ""):
        raise ValueError(f"{klass} missing 'mode' class attribute")
    _SOURCE_REGISTRY[klass.mode] = klass


def make_source(spec: Dict[str, Any]) -> DataSource:
    """spec dict → DataSource 인스턴스.

    spec 형식: {"mode": "<key>", "args": {<kwargs>}}
    """
    if not isinstance(spec, dict):
        raise TypeError(f"source spec must be dict, got {type(spec)}")
    mode = spec.get("mode")
    if mode not in _SOURCE_REGISTRY:
        raise ValueError(f"unknown source mode: {mode!r} (registered: {list(_SOURCE_REGISTRY)})")
    args = dict(spec.get("args", {}))
    klass = _SOURCE_REGISTRY[mode]
    return klass(**args)


# ─────────────────────────────────────────────────────────────────────
# Spec 헬퍼 (plan_maker에서 사용)
# ─────────────────────────────────────────────────────────────────────
def spec_recorded(paths: List[str]) -> Dict[str, Any]:
    return {"mode": "recorded", "args": {"paths": list(paths)}}


def spec_type_sampled(
    paths: List[str], n_items: int = 30, weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {"paths": list(paths), "n_items": int(n_items)}
    if weights is not None:
        args["weights"] = list(weights)
    return {"mode": "type_sampled", "args": args}


def spec_synthetic(bin_alias: Optional[str] = None, **cfg_kwargs) -> Dict[str, Any]:
    """SyntheticSource spec.

    Args:
        bin_alias: BIN_SPECS 키. 명시하면 SyntheticSource가 BIN_SPECS에서 bin_size 자동 set.
                   None이면 plan["bin"] 또는 SyntheticConfig 기본값 사용.
        **cfg_kwargs: SyntheticConfig 필드 (bin_size, max_items, min_item_mm, ...).
                      `bin_size` 명시하면 bin_alias 무시하고 그게 우선.
    """
    args: Dict[str, Any] = dict(cfg_kwargs)
    if bin_alias is not None:
        args["bin_alias"] = bin_alias
    return {"mode": "synthetic", "args": args}
