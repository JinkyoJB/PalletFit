from planning.heuristics.base import Base
from planning.heuristics.ZoneFit import ZoneFit
from planning.heuristics.ZoneFit.ZoneFunction import *
from planning.item import RotationType
from utils.get_value import *
from utils.position import get_horizontal_list, SameOrSmallerArea
from utils.checkPivot import checkPivot_R
from utils.pivot_generation import project_lines_left_to_pivots, getFrontLeftSideList, \
    project_lines_front_to_pivots, project_lines_down_to_pivots, collect_top_edge_candidate_positions
# from utils.painter.painter_plot import PainterPlot


import copy
from sortedcontainers import SortedKeyList


class OutlineFit(Base):
    """
    아이템 배치를 외곽(outer region)에 먼저 채우는 방식으로 진행하는 적재 알고리즘
    """
    def __init__(self, unfit_stop_setting=True, rotation_type=RotationType.BasicRotation):
        super().__init__(unfit_stop_setting, rotation_type)
        self.unfit_items = SortedKeyList(key=lambda item: (item.b_position[2], item.b_position[0], item.b_position[1]))

    def stack(self, bin, items_list):
        """
        bin에 items_list를 적재하되, composite(부모) 아이템도 생성될 수 있음.
        items_list는 원본 아이템들.
        """
        bk_bin_depth = copy.deepcopy(bin.depth)

        try2_bin_depth = bk_bin_depth + 10
        try3_bin_depth = bk_bin_depth + 20

        # 1) 원본 아이템의 ID 기준으로 index_map 생성
        index_map = {it._id: i for i, it in enumerate(items_list)}

        # 2) fit_result 초기화
        fit_result = [-1] * len(items_list)

        # 3) Composite 구성
        grouped_items_list = self.group_items_with_top(bin, items_list)
        grouped_items_list.sort(key=lambda x: (not x.is_composite, -(x.width * x.height)))

        # 4) 아이템 적재
        for comp_item in grouped_items_list:
            fitted, loaded_item = self.addItem(bin, comp_item)
            self._mark_fit_result(comp_item, fitted, fit_result, index_map)

        # 5) 실패한 remainder 재도전
        if fit_result.count(1) != len(items_list):
            remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            for item in remainder_items:
                fitted, loaded_item = self.addItem(bin, item)
                if fitted > 0:
                    print(f'{item.name}--------------remainder--------------')
                    self._mark_fit_result(item, fitted, fit_result, index_map)

            # # 그래도 실패 → bin 깊이 20mm 증가 후 재도전
            # remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            # if fit_result.count(1) != len(items_list):
            #     bin.depth = try2_bin_depth
            #     for item in remainder_items:
            #         fitted, loaded_item = self.addItem(bin, item)
            #         if fitted > 0:
            #             print(f'{item.name}--------------remainder 10mm up--------------')
            #             self._mark_fit_result(item, fitted, fit_result, index_map)
            # bin.depth = bk_bin_depth
            
            # # 그래도 실패 → bin 깊이 40mm 증가 후 재도전
            # remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            # if fit_result.count(1) != len(items_list):
            #     bin.depth = try3_bin_depth
            #     for item in remainder_items:
            #         fitted, loaded_item = self.addItem(bin, item)
            #         if fitted > 0:
            #             print(f'{item.name}--------------remainder 50mm up--------------')
            #             self._mark_fit_result(item, fitted, fit_result, index_map)
            # bin.depth = bk_bin_depth


            # # 그래도 실패 -> surface_ratio 0.6, bin 깊이 45mm 로 하여 호출
            # bk_surface_ratio = bin.support_surface_ratio
            # bin.depth = try2_bin_depth
            # bin.support_surface_ratio = 0.7
            # remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            # if fit_result.count(1) != len(items_list):
            #     for item in remainder_items:
            #         fitted, loaded_item = self.addItem(bin, item)
            #         if fitted > 0:
            #             print(f'{item.name}--------------bk_surface_ratio1--------------')
            #             self._mark_fit_result(item, fitted, fit_result, index_map)
            #     bin.support_surface_ratio = bk_surface_ratio
            # bin.depth = bk_bin_depth

            # # 그래도 실패 -> surface_ratio 0.6, bin 깊이 45mm 로 하여 호출
            # bk_surface_ratio = bin.support_surface_ratio
            # bin.depth = try3_bin_depth
            # bin.support_surface_ratio = 0.7
            # remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            # if fit_result.count(1) != len(items_list):
            #     for item in remainder_items:
            #         fitted, loaded_item = self.addItem(bin, item)
            #         if fitted > 0:
            #             print(f'{item.name}--------------bk_surface_ratio2--------------')
            #             self._mark_fit_result(item, fitted, fit_result, index_map)
            #     bin.support_surface_ratio = bk_surface_ratio
            # bin.depth = bk_bin_depth

            # 실패한 리스트 self.unfit_items에 추가
            for i, res in enumerate(fit_result):
                if res == 0:
                    self.unfit_items.add(items_list[i])

        bin.depth = bk_bin_depth
        return fit_result


    def addItem(self, bin, item, test=False):

        def _set_priority(item, num):
            item.priority = num
            return num
        
        # ----------------------------------------------
        rotations = [item.rotation_quat]
        alt_rt = RotationType.get_rotation_pair(item.rotation_quat)
        if alt_rt is not None:
            rotations.append(alt_rt)
        # ----------------------------------------------  

        # bin에 저장된 아이템이 없을 경우
        if bin.origin_item_id is None:
            init_pivot = [0, 0, 0]
            # 시도할 회전 타입 목록
            for init_rt in rotations:
                fitted, loaded_item = checkPivot_R(bin, item, init_pivot, init_rt)
                if fitted > 0:
                    if test:
                        return True, None
                    num = _set_priority(loaded_item, 1)
                    bin.store(loaded_item)
                    # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
                    loaded_item.update_child_positions_z()
                    print(f'{item.name}[ step {num} ] init rt {init_rt} 📥️')
                    return True, loaded_item

            
        # horizontal하게 배치하여 평평한 면을 만들 수 있는 배치 시도
        # horizontal_candidates = self.getHorizontalPivotList(bin, item)
        horizontal_candidates = get_horizontal_list(bin, item)

        for candidate_item in horizontal_candidates:
            get_direction_overlap(candidate_item, bin)

        horizontal_candidates_filtered = []
        for cand_item in horizontal_candidates:
            # 조건을 만족하는 방향의 개수
            pass_count = 0

            for dir_ in ['left','right','front','back']:
                gap, ratioA, ratioB, sum_overlap = cand_item.options['direction_overlap'][dir_]

                if dir_ in ('left','right'):
                    # gap ≤ bin.width*0.05 and ratio≥0.5
                    if gap <= bin.width*0.05 and ratioA >= 0.5:
                        pass_count += 1
                else:  # 'front','back'
                    # gap ≥ bin.height*0.05 and ratio≥0.5
                    if gap <= bin.height*0.05 and ratioA >= 0.5:
                        pass_count += 1

            # 4개 방향 중 3면 이상 만족 => 통과
            if pass_count >= 3:
                horizontal_candidates_filtered.append(cand_item)

        if horizontal_candidates_filtered:
            horizontal_candidates_filtered = sorted(
                horizontal_candidates_filtered,
                key=lambda item: min(item.options['direction_overlap'][d][0] for d in ['left','right','front','back'])
            )
            horizontal_item = horizontal_candidates_filtered[0]
            num = _set_priority(horizontal_item, 2)
            bin.store(horizontal_item)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            horizontal_item.update_child_positions_z()
            print(f'{item.name} [ step {num} ] Horizontal side 📥️')
            return True, horizontal_item


        # bin 저장된 아이템들 중, item과 같거나 작은 아이템을 찾아서 반환
        same_candidates, smaller_candidates = SameOrSmallerArea(bin, item)
        if same_candidates:
            same_candidates.sort(key=lambda x: (x.b_position[2], x.b_position[0], x.b_position[1]))
            same_item = same_candidates[0]
            num = _set_priority(same_item, 4)
            bin.store(same_item)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            same_item.update_child_positions_z()
            print(f'{same_item.name} [ step {num} ] Same area 📥️')
            return True, same_item
        
        # bin의 외각(앞쪽 또는 왼쪽)에 배치 시도
        FrontLightSideList = getFrontLeftSideList(bin, item)
        # 두 그룹의 후보들을 합치는데 둘중 None이면 제외
        FrontLightSideList = [x for x in FrontLightSideList if x is not None]

        if FrontLightSideList:
            for candidate_item in FrontLightSideList:
                get_direction_overlap(candidate_item, bin)

            # 각 후보의 'direction_overlap' 값 중 최소 gap을 기준으로 정렬
            FrontLightSideList.sort(key=lambda item: (item.b_position[2], 
                                                                    -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['left'][1]),    # overlap ratio A
                                                                    -(item.options['direction_overlap']['right'][1] + item.options['direction_overlap']['back'][1]),
                                                                    -item.options['direction_overlap']['front'][3], # overlap length
                                                                    -item.options['direction_overlap']['left'][3],  # overlap length
                                                                    item.options['direction_overlap']['front'][0],  # overlap gap
                                                                    item.options['direction_overlap']['left'][0],   # overlap gap
                                                                    item.options['direction_overlap']['right'][0],  # overlap gap
                                                                    item.options['direction_overlap']['back'][0]))  # overlap gap
            selected_candidate = FrontLightSideList[0]
            num = _set_priority(selected_candidate, 3)
            bin.store(selected_candidate)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            selected_candidate.update_child_positions_z()
            print(f'{selected_candidate.name} [ step {num} ] Front or Right side 📥️')
            return True, selected_candidate
        
        if smaller_candidates:
            smaller_candidates.sort(key=lambda x: (x.b_position[2], x.b_position[0], x.b_position[1]))
            smaller_item = smaller_candidates[0]
            num = _set_priority(smaller_item, 5)
            bin.store(smaller_item)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            smaller_item.update_child_positions_z()
            print(f'{smaller_item.name} [ step {num} ] Smaller area 📥️')
            return True, smaller_item
        
        projected_candidates = set()
        
        # ⬅️방향으로 사영한 pivot에 배치 시도
        left_cand_pivots = project_lines_left_to_pivots(bin)        

        
        # ⬇️방향으로 사영한 pivot에 배치 시도
        front_cand_pivots = project_lines_front_to_pivots(bin)
        

        down_cand_pivots = project_lines_down_to_pivots(bin)

        projected_candidates = left_cand_pivots + front_cand_pivots + down_cand_pivots

        # projected_candidates의 모든 요소의 요소를 np.float64로 변환
        pivots = self.unique_pivots(projected_candidates)

        projected_candidates = []      
        for rt in rotations:
            for pivot_pos in pivots:
                test_item = copy.deepcopy(item)
                test_item.rotation_quat = rt
                fitted, loaded_item = checkPivot_R(bin, test_item, [pivot_pos.x, pivot_pos.y, pivot_pos.z], rt)
                if fitted > 0:
                    projected_candidates.append(loaded_item)


        if projected_candidates:
            projected_candidates_filtered = []
            
            for cand_item in projected_candidates:
                # get_direction_overlap()를 호출하여, 'left','right','front','back' 방향의 (gap, ratio) 정보를 채운 후 반환
                gap_info = get_direction_overlap(cand_item, bin)
                
                pass_count = 0
                for dir_ in ['left','right','front','back']:
                    gap, ratioA, ratioB, sum_overlap = gap_info[dir_]
                    if dir_ in ('left','right'):
                        # 조건: gap ≤ bin.width * 0.05 and ratio ≥ 0.5
                        if gap <= bin.width * 0.05 and ratioA >= 0.5:
                            pass_count += 1
                    else:  # 'front','back'
                        # 조건: gap ≥ bin.height * 0.05 and ratio ≥ 0.5
                        if gap >= bin.height * 0.05 and ratioA >= 0.5:
                            pass_count += 1
                
                if pass_count >= 2:
                    projected_candidates_filtered.append(cand_item)

        
            # projected_candidates_filtered를 gap이 작은 순으로 정렬
            if projected_candidates_filtered:
                if 'A' in bin.name:
                    sort_key = lambda item: (
                        item.b_position[2],
                        -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['left'][1]),    # overlap ratio A
                        -(item.options['direction_overlap']['front'][3] + item.options['direction_overlap']['left'][3]),    # overlap length
                        item.b_position[1],
                        item.b_position[0],
                        -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['right'][1]),   # overlap ratio A
                        -(item.options['direction_overlap']['front'][3] + item.options['direction_overlap']['right'][3]),   # overlap length
                        item.options['direction_overlap']['front'][0],  # overlap gap
                        item.options['direction_overlap']['left'][0],   # overlap gap
                        item.options['direction_overlap']['right'][0],  # overlap gap
                        item.options['direction_overlap']['back'][0]    # overlap gap
                    )
                else:  # 'B'가 포함된 경우
                    sort_key = lambda item: (
                        item.b_position[2],
                        -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['left'][1]),    # overlap ratio A
                        -(item.options['direction_overlap']['front'][3] + item.options['direction_overlap']['left'][3]),    # overlap length
                        item.b_position[1],
                        -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['right'][1]),   # overlap ratio A
                        -(item.options['direction_overlap']['front'][3] + item.options['direction_overlap']['right'][3]),   # overlap length
                        item.options['direction_overlap']['front'][0],  # overlap gap
                        item.options['direction_overlap']['left'][0],   # overlap gap
                        item.options['direction_overlap']['right'][0],  # overlap gap
                        item.options['direction_overlap']['back'][0]    # overlap gap
                    )

                projected_candidates_filtered.sort(key=sort_key)

                projected_item = projected_candidates_filtered[0]
                num = _set_priority(projected_item, 4)
                bin.store(projected_item)
                # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
                projected_item.update_child_positions_z()
                print(f'{projected_item.name} [ step {num} ] Projected 📥️')
                return True, projected_item
        
             
        # horizontal_candidates가 있다면 배치 시도
        if horizontal_candidates:
            horizontal_candidates.sort(key=lambda x: (x.b_position[2], x.b_position[0], x.b_position[1]))
            horizontal_item2 = horizontal_candidates[0]
            num = _set_priority(horizontal_item2, 4)
            bin.store(horizontal_item2)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            horizontal_item2.update_child_positions_z()
            print(f'{horizontal_item2.name} [ step {num} ] Horizontal2 📥️')
            return True, horizontal_item2
        
        # projected_candidates가 있다면 배치 시도
        if projected_candidates:
            if 'A' in bin.name:
                sort_key = lambda item: (
                    item.b_position[2],
                    -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['left'][1]),    # overlap ratio A
                    -(item.options['direction_overlap']['front'][3] + item.options['direction_overlap']['left'][3]),    # overlap length
                    -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['right'][1]),   # overlap ratio A
                    -(item.options['direction_overlap']['front'][3] + item.options['direction_overlap']['right'][3]),   # overlap length
                    # item.b_position[1],  # y 우선
                    # item.b_position[0],
                    # -(item.options['direction_overlap']['right'][1] + item.options['direction_overlap']['back'][1]),
                    item.options['direction_overlap']['front'][0],  # overlap gap
                    item.options['direction_overlap']['left'][0],   # overlap gap
                    item.options['direction_overlap']['right'][0],  # overlap gap
                    item.options['direction_overlap']['back'][0]    # overlap gap
                )
            else:  # 'B'가 포함된 경우
                sort_key = lambda item: (
                    item.b_position[2],
                    -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['left'][1]),    # overlap ratio A
                    -(item.options['direction_overlap']['front'][3] + item.options['direction_overlap']['left'][3]),    # overlap length
                    -(item.options['direction_overlap']['front'][1] + item.options['direction_overlap']['right'][1]),   # overlap ratio A
                    -(item.options['direction_overlap']['front'][3] + item.options['direction_overlap']['right'][3]),   # overlap length
                    # item.b_position[1],  # y 우선
                    # item.b_position[0],
                    # -(item.options['direction_overlap']['right'][1] + item.options['direction_overlap']['back'][1]),
                    item.options['direction_overlap']['front'][0],  # overlap gap
                    item.options['direction_overlap']['left'][0],   # overlap gap
                    item.options['direction_overlap']['right'][0],  # overlap gap
                    item.options['direction_overlap']['back'][0]    # overlap gap
                )
            projected_candidates.sort(key=sort_key)
            projected_item2 = projected_candidates[0]
            num = _set_priority(projected_item2, 4)
            bin.store(projected_item2)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            projected_item2.update_child_positions_z()
            print(f'{projected_item2.name} [ step {num} ] Projected2 📥️')
            return True, projected_item2

        
        # _ffd 호출하여 모든 아이템의 4방향 배치 시도
        fitted, loaded_item = self._ffd(bin, item, test)
        if fitted > 0:
            num = _set_priority(loaded_item, 7)
            bin.store(loaded_item)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            loaded_item.update_child_positions_z()
            print(f'{loaded_item.name} [ step {num} ] FFD 📥️')
            return True, loaded_item

        # 위쪽에 배치 시도
        UpSide_candidates = []
        UpSide_pivots = collect_top_edge_candidate_positions(bin, item)
        if len(UpSide_pivots) > 0:
            for available_pivot in UpSide_pivots:
                for rt in rotations:
                    fitted, loaded_item = checkPivot_R(bin, item, available_pivot, rt, apply_margin=False)
                    if fitted > 0:
                        UpSide_candidates.append(loaded_item)

        if len(UpSide_candidates) > 0:
            UpSide_candidates.sort(key=lambda x: x._bottom_overlap_area, reverse=True)
            UpSide_item = UpSide_candidates[0]
            num = _set_priority(UpSide_item, 9)
            bin.store(bin, UpSide_item)
            # 상단merge를 적용하는 휴리스틱 인 경우, 자식 아이템들의 z 위치 갱신
            UpSide_item.update_child_positions_z()
            print(f'{UpSide_item.name} [ step {num} ] Up side2 📥️')
            return True, UpSide_item
        
        # # zonefit 호출하여 bin의 위쪽에 배치 시도
        # fitted = self._zoneFit(bin, item)
        # if fitted > 0:
        #     num = _set_priority(loaded_item, 11)
        #     print(f'{loaded_item.name} [ step {num} ] ZoneFit 📥️')
        #     loaded_item = global_item_manager.get(item._id)
        #     return True, loaded_item
        
        return False, None


    
    def _ffd(self, bin, item, test):
        '''
        내부에서 self.fixPivot을 호출하는 FFD 휴리스틱
        '''
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
        
        item_in_bin = bin.get_all_items()
        item_in_bin = sorted(item_in_bin, key=lambda ib: (ib.b_position[2], distance_sq_from_origin(ib)))

        # (3) 아이템 -> 축 -> 회전 순
        for ib in item_in_bin:
            w, h, d = ib.getDimension()
            ib_x, ib_y, ib_z = ib.b_position
            margin_x = bin.margin_x
            margin_y = bin.margin_y

            for axis in range(0, 6):
                for rt in rotate:
                    copy_item = copy.deepcopy(item)
                    copy_item.rotation_quat = rt
                    iw, _, _ = copy_item.getDimension()

                    if axis == 0: # 앞측 꼭지점 기준
                        pivot = [ib_x,
                                ib_y + h + margin_y,
                                ib_z]
                    elif axis == 1:  # 우측 꼭지점 기준
                        pivot = [ib_x + w + margin_x,
                                ib_y,
                                ib_z]
                    elif axis == 2:  # 앞측 꼭지검 기준2
                        pivot=[ib.ex - iw - margin_x,
                                ib_y + h + margin_y,
                                ib_z]
                    elif axis == 3: # 좌측 꼭지점 기준
                        pivot = [ib_x - iw - margin_x,
                                ib_y,
                                ib_z]
                    elif axis == 4: 
                        pivot = [ib.ex - iw,
                                ib.ey + margin_y,
                                ib_z]
                    elif axis == 5:
                        pivot = [
                            ib.ex - iw - margin_x,
                            ib_y,
                            ib_z
                        ]
                    
                    if axis in [0, 2, 3]:
                        # 배치 시도
                        fitted, loaded_item = checkPivot_R(bin, item, pivot, rt)
                        if fitted > 0:
                            if test:
                                return fitted, None
                            return fitted, loaded_item
                    else:
                        fitted, loaded_item = checkPivot_R(bin, item, pivot, rt, apply_margin=False)
                        if fitted > 0:
                            if test:
                                return fitted, None
                            return fitted, loaded_item

        return False, None

    def _zoneFit(self, bin, item):
        """
        ZoneFit 휴리스틱을 적용하는 메서드
        """
        # painter = PainterPlot(bin)

        zonefit = ZoneFit(
                unfit_stop_setting = self._unfit_stop_setting, 
                rotation_quat = self._rotation_type,
            )
        
        fit_result = zonefit.stack(bin, [item])
        fitted = fit_result[0]
        return fitted

    def unique_pivots(self, pivots, tol=1e-2):
        """
        내부 헬퍼 함수:
        - pivots 리스트에서 중복된 pivot 제거
        - 허용 오차 tol을 사용하여 중복 제거
        """
        uniq_keys, uniq_pivots = set(), []
        for pv in pivots:
            key = (float(pv.x), float(pv.y), float(pv.z), tuple(map(float, pv.rt)))
            if key not in uniq_keys:
                uniq_keys.add(key)
                uniq_pivots.append(pv)

        return uniq_pivots