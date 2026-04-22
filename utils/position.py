
# utils/position.py
from planning.item import RotationType

from .get_value import get_dim_for_rt
from .checkPivot import checkPivot_R


def get_horizontal_list(bin, item):
    '''
    'search item'의 좌, 우, 앞, 뒤에 배치했을 때, item의 top(ez)높이와 동일하면서, 
    'search item'의 상단에 아이템이 배치되어 있지 않은 경우의 candidates를 반환.

    'search item'의 상단이 빈경우를 탐색하는 이유는 horizontal을 고려하는 이유가 평평한 면을 
    만들기 위함이기 때문.

    return: candidates
    '''
    candidates = []
    group_idx = RotationType.get_BasicRotation_index(item.rotation_quat)
    rotate = RotationType.BasicRotation[group_idx]

    for item_in_bin in bin.get_all_items():
        for rt in rotate:
            item_w, item_h, item_d = get_dim_for_rt(item, rt)

            # 좌, 우, 앞, 뒤 방향에 대한 위치
            directions = [
                ('right', [item_in_bin.ex + bin.margin_x, item_in_bin.b_position[1], item_in_bin.b_position[2]]),
                ('left', [item_in_bin.b_position[0] - item_w - bin.margin_x, item_in_bin.b_position[1], item_in_bin.b_position[2]]),
                ('back', [item_in_bin.b_position[0], item_in_bin.ey + bin.margin_y, item_in_bin.b_position[2]]),
                ('front', [item_in_bin.b_position[0], item_in_bin.b_position[0] - item_h -bin.margin_y, item_in_bin.b_position[2]])
            ]

            for direction, (x, y, z) in directions:
                fitted, loaded_item = checkPivot_R(bin, item, [x, y, z], rt)
                if (fitted>0) and loaded_item.ez == item_in_bin.ez:
                        candidates.append(loaded_item)

    return candidates


def SameOrSmallerArea(bin, item):
    '''
    조건 1: 면적이 거의 동일한 경우 (오차를 각각 bin.margin_x, bin.margin_y 이하로 허용)
    조건 2: candidate가 더 작은 경우
    조건1,2를 만족하는 후보들을 반환
    '''
    same_candidates = []
    smaller_candidates = []

    # 가능한 회전 방향
    group_idx = RotationType.get_BasicRotation_index(item.rotation_quat)
    rotate = RotationType.BasicRotation[group_idx]

    for ref_item in bin.get_visible_items_topdown():
        rw, rh, rd = ref_item.getDimension()

        for rt in rotate:
            iw, ih, id = get_dim_for_rt(item, rt)

            # 조건 1: 면적이 거의 동일한 경우 (오차를 각각 bin.margin_x, bin.margin_y 이하로 허용)
            x_range = 2*bin.margin_x
            y_range = 2*bin.margin_y
            cond_same_area = (abs(rw - iw) <= x_range) and \
                            (abs(rh - ih) <= y_range)
            # 조건 2: candidate가 더 작은 경우
            cond_smaller = (iw < rw) and (ih < rh)

            if cond_same_area:
                fitted, loaded_item = checkPivot_R(bin, item, [ref_item.b_position[0], ref_item.b_position[1], ref_item.ez], rt)
                if fitted > 0:
                    same_candidates.append(loaded_item)
            if cond_smaller:
                fitted, loaded_item = checkPivot_R(bin, item, [ref_item.b_position[0], ref_item.b_position[1], ref_item.ez], rt)
                if fitted > 0:
                    smaller_candidates.append(loaded_item)

    return same_candidates, smaller_candidates