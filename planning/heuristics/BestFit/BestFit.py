from planning.item import RotationType
from planning.heuristics.base import Base
from utils.get_value import get_direction_overlap, distance_sq_from_origin

from utils.position import checkPivot_R


class BestFit(Base):
    '''
    동적지표를 기준으로 우선순위가 높은 pivot을 선택하여 적재하는 알고리즘
    '''
    def __init__(self, unfit_stop_setting=True, rotation_type=RotationType.BasicRotation):
        super().__init__(unfit_stop_setting, rotation_type)
    
    def addItem(self, bin, item, test=False):
        self.set_init_PT(bin, bin.get_all_items())

        feasible_pivots = []
        for pivot in bin.pivotTree.in_order_traversal():
            fitted, loaded_item = checkPivot_R(bin, item, [pivot.x, pivot.y, pivot.z], pivot.rt)
            if fitted > 0:
                feasible_pivots.append((pivot, loaded_item))

        if not feasible_pivots:
            return False, None

        def pivot_sort_key(entry):
            pivot, loaded_item = entry
            # 동적지표 계산(옵션 키 없을 수 있으니 방어적 접근)
            get_direction_overlap(loaded_item, bin)
            dg = loaded_item.options.get('direction_overlap', {})
            front = dg.get('front', [0, 0, 0, 0])
            left  = dg.get('left',  [0, 0, 0, 0])
            bottom_overlap = getattr(loaded_item, '_bottom_overlap_area', 0.0)

            # 파이썬 정렬은 오름차순 → 큰값을 우선하려면 -를 붙임
            return (
                pivot.z,                                      # 낮은 z 먼저
                -round(bottom_overlap, 2),                    # 바닥 접촉면이 클수록 우선
                -(front[1] + left[1]),                        # overlap area(가정) 클수록 우선
                -(front[3] + left[3]),                        # overlap length(가정) 클수록 우선
                distance_sq_from_origin(loaded_item),         # 원점에서 가까울수록 우선
            )

        feasible_pivots.sort(key=pivot_sort_key)
        best_pivot, best_loaded_item = feasible_pivots[0]

        bin.store(best_loaded_item)
        # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
        best_loaded_item.update_child_positions_z()
        self.store2Pivot(bin)
        return True, best_loaded_item


