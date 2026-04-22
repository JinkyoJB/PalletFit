from dataclasses import dataclass
from typing import List, Tuple, Optional
from planning.item import RotationType
from planning.heuristics.base import Base
from utils.position import checkPivot_R
import math

@dataclass
class PlanSignature:
    bin_w: float; bin_h: float; bin_d: float
    m_x: float; m_y: float
    item_w: float; item_h: float; item_d: float
    rotation_type: RotationType

    @classmethod
    def from_state(cls, bin, item, rotation_type):
        return cls(
            bin.width, bin.height, bin.depth,
            getattr(bin, "margin_x", 0.0), getattr(bin, "margin_y", 0.0),
            item.width, item.height, item.depth,
            rotation_type
        )

class SingleFit(Base):
    """
    동일 규격 아이템을 한 개 Bin에 격자로 최대 적재.
    기본 배치(RT_WHD) vs 90도 회전(RT_HWD) 두 케이스 비교,
    기본 케이스에서 X-스트립 보너스 채우기까지 고려.
    """
    def __init__(self, unfit_stop_setting=True, rotation_type=RotationType.BasicRotation, verbose: bool=False):
        super().__init__(unfit_stop_setting, rotation_type)
        self.placement_list: List[Tuple[float, float, float, RotationType]] = []
        self.max_item_num: int = 0
        self.main_rotation_type: Optional[int] = None  # 0: WHD, 1: HWD
        self.main_nx = self.main_ny = self.main_nz = 0
        self.final_left_len_x = 0.0
        self.final_left_len_y = 0.0
        self._plan_sig: Optional[PlanSignature] = None
        self.verbose = verbose

    def __repr__(self):
        return self.__class__.__name__

    # ---------- helpers ----------
    @staticmethod
    def _grid_capacity(length: float, item: float, margin: float) -> Tuple[int, float]:
        """Return (n, leftover) for 1D packing with margin between consecutive items."""
        if item <= 0:
            return 0, length
        step = item + max(0.0, margin)
        if step <= 0:
            return 0, length
        if length < item:
            return 0, length
        n = 1 + math.floor((length - item) / step)
        n = max(0, n)
        used = item + (n - 1) * step if n > 0 else 0.0
        leftover = max(0.0, length - used)
        return n, leftover

    def _need_rebuild(self, bin, item) -> bool:
        sig = PlanSignature.from_state(bin, item, self._rotation_type)
        if self._plan_sig != sig or not self.placement_list:
            self._plan_sig = sig
            return True
        return False

    def addItem(self, bin, item):
        """
        하나의 bin에 대해 item을 1개 적재 시도.
        성공 시 bin.store(loaded_item) 수행.
        return: (fitted: bool/int, loaded_item or None)
        """
        # 계획 필요시 재생성
        if self._need_rebuild(bin, item):
            built = self.make_placement_list(bin, item)
            if not built:
                bin.unfit_items.append(item)
                return False, None

        # 현재까지 적재된 개수가 다음 인덱스
        idx = getattr(bin, "size", len(getattr(bin, "items", [])))
        if idx >= self.max_item_num:
            bin.unfit_items.append(item)
            return False, None

        x, y, z, r = self.placement_list[idx]
        fitted, loaded_item = checkPivot_R(bin, item, (x, y, z), r)
        if fitted > 0:
            bin.store(loaded_item)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            loaded_item.update_child_positions_z()
            return fitted, loaded_item
        else:
            # 해당 피벗에서 실패하면 이번 전략은 한계 → unfit
            bin.unfit_items.append(item)
            return False, None

    # ---------- plan builders ----------
    def make_placement_list(self, bin, item) -> bool:
        """
        배치 계획 생성.
        두 케이스 비교:
          - case1: RT_WHD (width=x, height=y)
          - case2: RT_HWD (height=x, width=y)
        case1가 선택되면 X 여분 띠에 회전 배치 보너스 고려.
        """
        self.placement_list.clear()
        self.max_item_num = 0
        self.main_rotation_type = None
        self.main_nx = self.main_ny = self.main_nz = 0
        self.final_left_len_x = self.final_left_len_y = 0.0

        # 기본 높이(=bin.depth) 체크
        if bin.depth < item.depth:
            if self.verbose:
                print("[SingleFit] 아이템 높이(item.depth)가 bin.depth보다 큽니다.")
            return False

        # 공통 Z 용량
        nz, _ = self._grid_capacity(bin.depth, item.depth, 0.0)
        if nz == 0:
            return False

        # case1: RT_WHD
        c1_nx, c1_left_x = self._grid_capacity(bin.width,  item.width,  bin.margin_x)
        c1_ny, c1_left_y = self._grid_capacity(bin.height, item.height, bin.margin_y)
        c1 = c1_nx * c1_ny * nz

        # case2: RT_HWD (x<->y에 item.width/height 스왑)
        c2_nx, c2_left_x = self._grid_capacity(bin.width,  item.height, bin.margin_x)
        c2_ny, c2_left_y = self._grid_capacity(bin.height, item.width,  bin.margin_y)
        c2 = c2_nx * c2_ny * nz

        if c1 == 0 and c2 == 0:
            return False

        # case1 보너스 스트립(남은 X 띠에 RT_HWD로 채우기)
        c1_bonus = 0
        b_nx = b_ny = 0
        if c1 > 0 and c1_left_x >= item.height:
            # 보너스 영역 너비 = c1_left_x
            b_nx, _ = self._grid_capacity(c1_left_x, item.height, bin.margin_x)
            b_ny, _ = self._grid_capacity(bin.height, item.width, bin.margin_y)
            c1_bonus = b_nx * b_ny * nz

        # 총합 비교
        c1_total = c1 + c1_bonus
        pick_case1 = (c1_total >= c2)

        if pick_case1:
            # 메인: RT_WHD
            self.main_rotation_type = 0
            self.main_nx, self.main_ny, self.main_nz = c1_nx, c1_ny, nz
            self.final_left_len_x = c1_left_x
            # y-여분은 "두 배치 중 더 큰 여유"로 대략 보고
            self.final_left_len_y = c1_left_y

            # 메인 격자 피벗
            for k in range(nz):
                for i in range(c1_nx):
                    for j in range(c1_ny):
                        x = i * (item.width  + bin.margin_x)
                        y = j * (item.height + bin.margin_y)
                        z = k * item.depth
                        self.placement_list.append((x, y, z, RotationType.RT_WHD))

            # 보너스 띠 (RT_HWD)
            if c1_bonus > 0:
                base_x = c1_nx * (item.width + bin.margin_x)
                for k in range(nz):
                    for i in range(b_nx):
                        for j in range(b_ny):
                            x = base_x + i * (item.height + bin.margin_x)
                            y = j * (item.width  + bin.margin_y)
                            z = k * item.depth
                            self.placement_list.append((x, y, z, RotationType.RT_HWD))

        else:
            # 메인: RT_HWD
            self.main_rotation_type = 1
            self.main_nx, self.main_ny, self.main_nz = c2_nx, c2_ny, nz
            self.final_left_len_x = c2_left_x
            self.final_left_len_y = c2_left_y

            for k in range(nz):
                for i in range(c2_nx):
                    for j in range(c2_ny):
                        x = i * (item.height + bin.margin_x)
                        y = j * (item.width  + bin.margin_y)
                        z = k * item.depth
                        self.placement_list.append((x, y, z, RotationType.RT_HWD))

        self.max_item_num = len(self.placement_list)
        if self.verbose:
            print(f"[SingleFit] case1(WHD)+bonus={c1_total}, case2(HWD)={c2}, picked={'case1' if pick_case1 else 'case2'}")
            print(f"[SingleFit] placements={self.max_item_num}")
        return self.max_item_num > 0
