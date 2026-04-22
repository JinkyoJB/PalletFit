# planning/BinSpecsDict.py
from typing import Dict, List, Literal, TypedDict
from planning.item import RotationType  # 기존 스펙에 RotationType.* 참조가 있으므로

class BinSpec(TypedDict, total=False):
    name: str
    width: int
    height: int
    depth: int
    max_weight: int
    w_position: List[float]        # [x, y, z] (world 좌표, m)
    rotation_quat: List[float] | RotationType  # [w,x,y,z] or enum
    margin_x: float | int
    margin_y: float | int
    options: dict
    unit: Literal["mm", "cm", "m"]
    corner: int
    support_surface_ratio: float
    # 필요한 키를 더 추가해도 OK

#quaternion: List[float]  # [x, y, z, w] (수정됨)
BIN_SPECS: Dict[str, BinSpec] = {
    "default1": {
        "name": "default1",
        "width": 500, "height": 500, "depth": 485,
        "max_weight": 99999,
        "w_position": [0.36, -0.315, 0.006],
        # 기존 [1, 0, 0, 0] -> [0, 0, 0, 1]
        "rotation_quat": [0.0, 0.0, 0.0, 1.0], 
        "margin_x": 6, "margin_y": 6,
        "options": {"min_bin": [400, 400, 400]},
        "unit": "mm", "corner": 0, "support_surface_ratio": 0.6,
    },
    "default2": {
        "name": "default2",
        "width": 500, "height": 500, "depth": 400,
        "max_weight": 99999,
        "w_position": [0.41+0.5, -0.315+0.5, 0.006],
        # 기존 [0, 0, 0, -1] -> [0, 0, -1, 0]
        "rotation_quat": [0.0, 0.0, -1.0, 0.0],
        "margin_x": 7, "margin_y": 7,
        "options": {"min_bin": [400, 400, 400]},
        "unit": "mm", "corner": 0, "support_surface_ratio": 0.65,
    },
    "experiment": {
        "name": "experiment",
        "width": 500, "height": 500, "depth": 485,
        "max_weight": 99999,
        "w_position": [0.36+0.5, -0.315+0.5, 0.006],
        # 기존 [0, 0, 0, -1] -> [0, 0, -1, 0]
        "rotation_quat": [0.0, 0.0, -1.0, 0.0],
        "margin_x": 5, "margin_y": 5,
        "options": {"min_bin": [400, 400, 400]},
        "unit": "mm", "corner": 0, "support_surface_ratio": 0.5,
    },
    "A": {
        "name": "bin_A",
        "width": 485, "height": 326, "depth": 140,
        "max_weight": 99999,
        "w_position": [-0.234, 0.6725, 0.001 + 1e-6],
        # RotationType.RT_HWD 값 자체를 이전에 수정한 대로 사용하시면 됩니다.
        "rotation_quat": RotationType.RT_HWD,
        "margin_x": 8, "margin_y": 8,
        "options": {"fit_items": []},
        "unit": "mm", "corner": 0, "support_surface_ratio": 0.9,
    },
    "B": {
        "name": "bin_B",
        "width": 485, "height": 322, "depth": 140,
        "max_weight": 99999,
        "w_position": [0.34875, -0.315, 0.006],
        "margin_x": 8, "margin_y": 8,
        "options": {"fit_items": []},
        "unit": "mm", "corner": 0, "support_surface_ratio": 0.9,
    },
    "SKT": {
        "name": "bin_SKT",
        "width": 1100, "height": 1100, "depth": 150,
        "max_weight": 99999,
        "w_position": [0.34875, -0.315, 0.006],
        # RotationType.RT_HWD 값 자체를 이전에 수정한 대로 사용하시면 됩니다.
        "rotation_quat": RotationType.RT_HWD,
        "margin_x": 10, "margin_y": 10,
        "options": {"min_bin": [1100, 1100, 1800]},
        "unit": "mm", "corner": 0, "support_surface_ratio": 0.5,
    },
    "experiment_RL": {
        "name": "experiment_RL",
        "width": 1000, "height": 1000, "depth": 1000,
        "max_weight": 99999,
        "w_position": [0.36+0.5, -0.315+0.5, 0.006],
        # 기존 [0, 0, 0, -1] -> [0, 0, -1, 0]
        "rotation_quat": [0.0, 0.0, -1.0, 0.0],
        "margin_x": 0.0, "margin_y": 0.0,
        "options": {"min_bin": [400, 400, 400]},
        "unit": "mm", "corner": 0, "support_surface_ratio": 0.5,
    },
    "experiment_RL2": {
        "name": "experiment_RL2",
        "width": 100, "height": 100, "depth": 100,
        "max_weight": 99999,
        "w_position": [0.36+0.5, -0.315+0.5, 0.006],
        # 기존 [0, 0, 0, -1] -> [0, 0, -1, 0]
        "rotation_quat": [0.0, 0.0, -1.0, 0.0],
        "margin_x": 0.0, "margin_y": 0.0,
        "options": {"min_bin": [400, 400, 400]},
        "unit": "mm", "corner": 0, "support_surface_ratio": 0.5,
    },}

__all__ = ["BIN_SPECS", "BinSpec"]
