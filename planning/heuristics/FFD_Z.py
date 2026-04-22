from planning.heuristics.base import Base
from utils.item_utils import *
from planning.item import RotationType
from utils.position import checkPivot_R


class FFD_Z(Base):
    def __init__(self, unfit_stop_setting = True, rotation_type = RotationType.BasicRotation):
        super().__init__(unfit_stop_setting, rotation_type)

    def __repr__(self):
        return self.__class__.__name__
    
    def addItem(self, bin, item, test=False):
        '''
        하나의 bin에 대해 item을 적재, 
        적재 가능하면 bin.items에 추가, 불가능하면 bin.unfit_items에 추가
        return: 적재 가능 여부
        '''
        fitted = False
        # 🔥 rotation 후보 리스트 만들기
        if self._rotation_type == RotationType.All:
            rotate = RotationType.All

        elif self._rotation_type == RotationType.UnRotation:
            rotate = RotationType.UnRotation

        elif self._rotation_type == RotationType.BasicRotation:
            group_idx = RotationType.get_BasicRotation_index(item.rotation_quat)
            rotate = RotationType.BasicRotation[group_idx]

        else:
            raise ValueError(f"Unsupported rotation_type setting: {self._rotation_type}")
        
        if bin.origin_item_id is None:
            pivot = [0,0,0]
            for rt in rotate:
                fitted, loaded_item = checkPivot_R(bin, item, pivot, rt)
                if fitted > 0:
                    if test:
                        return fitted, None
                    bin.store(loaded_item)
                    loaded_item.update_child_positions_z()
                    self.store2Pivot(bin)
                else:
                    bin.unfit_items.append(item)
                return fitted, loaded_item
        
        z_group = bin.group_by_z()

        for z in sorted(z_group.keys()):
            item_list_at_z = z_group[z]
            for ib in item_list_at_z:
                ib_x, ib_y, ib_z = ib.b_position

                for axis in range(2):
                    for rt in rotate:
                        if axis == 0:
                            pivot = [
                                ib.ex + bin.margin_x,
                                ib_y,
                                ib_z,
                            ]
                        elif axis == 1:
                            pivot = [
                                ib_x,
                                ib.ey + bin.margin_y,
                                ib_z,
                            ]

                        fitted, loaded_item = checkPivot_R(bin, item, pivot, rt)
                        if fitted > 0:
                            if test:
                                return fitted, None
                            bin.store(loaded_item)
                            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
                            loaded_item.update_child_positions_z()
                            return fitted, loaded_item
        
        # 새로운 층에 적재
        for z in sorted(z_group.keys()):
            item_list_at_z = z_group[z]
            for ib in item_list_at_z:
                for rt in rotate:
                    pivot = [
                        ib_x,
                        ib_y,
                        ib.ez,
                    ]

                    fitted, loaded_item = checkPivot_R(bin, item, pivot, rt)
                    if fitted > 0:
                        if test:
                            return fitted, None
                        bin.store(loaded_item)
                        # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
                        loaded_item.update_child_positions_z()
                        return fitted, loaded_item


        if fitted < 0:
            bin.unfit_items.append(item)
            return fitted, None
        