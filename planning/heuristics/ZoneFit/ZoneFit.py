from planning.item import RotationType
from planning.heuristics.base import Base
from planning.heuristics.ZoneFit.ZoneAVLtree import ZoneAVLTree
from planning.heuristics.ZoneFit.ZoneFunction import *
from utils.item_utils import *
from planning.heuristics.FFD_M import FFD_M
from utils.position import checkPivot_R
from utils.overlap import compute_overlap_volume

import numpy as np
import copy


class ZoneFit(Base):
    '''
    zone을 나눠서 아이템을 적재하는 휴리스틱
    '''

    def __init__(self, unfit_stop_setting=True, rotation_type=RotationType.BasicRotation):
        super().__init__(unfit_stop_setting, rotation_type)

        # self.bin = None
        self.items_list = None
        self.global_flag = False

    def __repr__(self):
        return self.__class__.__name__
    
    def set_init_bin(self, item_list):
        '''
        기존에 담겨있는 아이템을 기준으로 zone분할
        '''
        self.tree = ZoneAVLTree()    # Zone 객체를 AVL 트리에 저장
        init_zone = get_init_zone(self.bin)
        self.tree.insert(init_zone)

        for item in item_list:
            intersects = self.tree.find_zones_that_intersect_item_and_corners(item)
            for zone, corner_list in intersects:
                # 1) 교차하는 zone을 트리에서 삭제
                self.tree.delete(zone)

                # 2) corner_list에서 "제일 우측상단에 있는 모서리"를 찾기
                #    예: z > y > x 순으로 가장 큰 corner를 우측상단이라고 가정
                top_right_corner = max(corner_list, key=lambda c: (c[2], c[1], c[0]))

                # 3) split_zone 준비
                #    dx, dy, dz = (top_right_corner - zone의 시작점)
                dx = top_right_corner[0] - zone.x
                dy = top_right_corner[1] - zone.y
                dz = top_right_corner[2] - zone.z

                # 4) zone을 8개 서브존으로 분할 (split_zone)
                splitted_zones = split_zone(zone, dx, dy, dz)

                # 5) 아이템 영역 sub-zone 은 다시 삽입하지 않음
                for sz in splitted_zones[1:]:
                    self.tree.insert(sz)
                merge_shared_face_zones(self.tree)

    def addItem(self, bin, item, test=False):
        """
        1) zone_list 후보들 각각에 대해
        - tree_copy, zone_copy 로 '가상 시뮬레이션' 실시
        - leftover, score 계산
        2) 모두 끝난 뒤, 'best zone' 선택
        3) 원본 self.tree 에 최종적으로 delete/insert 반영
        """
        candidates = []  # zone 후보

        # (1) leftover 충분한 zone들 찾기
        zone_list = self.tree.get_sorted_zones_zyx()  # item_leftover보다 큰 zone을 찾음
        if not zone_list:
            # print('적재 가능한 zone이 없습니다.(none zone_list)')
            return False, None
        
        # (2) 회전타입 후보 설정
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
        
        bin_faces = bin.update_face_cache()   #2,3,4,5가 앞,뒤,왼,오 

        # (3) 시뮬레이션 결과 저장 구조
        for zone in zone_list:
            for rt in rotate:
                # (A) checkPivot_R, xy_match 등 계산
                item.rotation_quat = rt

                zone_faces = zone.update_face_cache()
                coincident = find_coincident_faces_list(bin_faces, zone_faces)      # bin과 zone의 공통된 face 찾기
                margin = self._decide_margin(bin, coincident)                         # bin에 접한 면에 따라 margin 결정
                global_pivot = [zone.x + margin[0], zone.y + margin[1], zone.z + margin[2]]
                fitted, loaded_item = checkPivot_R(bin, item, global_pivot, rt)

                if fitted < 0:  # 적재 불가능한 경우
                    candidates.append({
                        'zone': zone,
                        'rt': rt,
                        'fitted_item': False,
                        'global_pivot': global_pivot,
                        'tree': False,
                        'xy_match': False,
                        'max_leftover': 0,
                        'min_leftover': np.inf,
                        'average_leftover': 0,
                        'diagonal_of_max_leftover': 0,
                        'outer_leftover': 0,
                        'vertices_after_placement': None
                    })
                    continue

                # (A) 시뮬레이션
                sim_result = self.simulate_zone_placement(bin, zone, global_pivot, loaded_item)

                tree_copy = sim_result['tree']
                is_xy_matched = sim_result['xy_match']
                max_leftover = sim_result['max_leftover']
                min_leftover = sim_result['min_leftover']
                average_leftover = sim_result['average_leftover']
                diagonal_of_max_leftover = sim_result['diagonal_of_max_leftover']
                outer_lf = sim_result['outer_leftover']
                vtx_after = sim_result['vertices_after_placement']

                # 후보 딕셔너리
                candidates.append({
                    'zone': zone,
                    'rt': rt,
                    'fitted_item': loaded_item,
                    'global_pivot': global_pivot,
                    'tree': tree_copy,
                    'xy_match': is_xy_matched,
                    'max_leftover': max_leftover,
                    'min_leftover': min_leftover,
                    'average_leftover': average_leftover,
                    'diagonal_of_max_leftover': diagonal_of_max_leftover,
                    'outer_leftover': outer_lf,
                    'vertices_after_placement': vtx_after,
                })

        # 4) fitted_item != False 인 것만 필터 => valid_candidates
        valid_candidates = [c for c in candidates if c['fitted_item'] is not False]
        if not valid_candidates:
            return False, None
        
        def get_best_candidate(valid_candidates):
            """
            valid_candidates에서 최적의 후보 선택
            """
            # xy_match가 True인 후보가 있는지 확인
            has_xy_match = any(c['xy_match'] for c in valid_candidates)

            if not has_xy_match:  
                # (B) xy_match가 모두 False면, zone.z == 0인 후보들만 선택 후 다시 정렬
                zero_z_candidates = [c for c in valid_candidates if c['zone'].z == 0]

                if zero_z_candidates:  # zone.z == 0 인 후보가 있다면
                    best_candidates = sorted(
                        zero_z_candidates,
                        key=lambda c: (
                            -c['outer_leftover'],  
                            c['vertices_after_placement'][1],         
                            c['zone'].y,                       
                            c['zone'].x,
                            c['vertices_after_placement'][0],         
                            c['diagonal_of_max_leftover'],  
                            -c['min_leftover'],  
                            c['max_leftover'],
                        ), 
                        reverse=False  # 오름차순 정렬
                    )
                    return best_candidates[0]

            best_candidates = sorted(
                valid_candidates,
                key=lambda c: (
                    -int(c['xy_match']),   # True=1 > False=0
                    c['zone'].z,
                    -c['outer_leftover'],  
                    c['vertices_after_placement'][1],         
                    c['zone'].y,                       
                    c['zone'].x,
                    c['vertices_after_placement'][0],         
                    c['diagonal_of_max_leftover'],  
                    -c['min_leftover'],  
                    c['max_leftover'],
                    ), 
                reverse=False  # 오름차순 정렬
            )
            return best_candidates[0]

        best = get_best_candidate(valid_candidates)

        if test:
            return True
        
        # item을 bin에 추가
        bin.store(best['fitted_item'])

        # self.tree업데이트
        self.tree = best['tree']

        # loaded_item (가상) 좌표 업데이트
        return True, best['fitted_item']                              
    
    def _decide_margin(self, bin, coincident):
        """
        기존의 if-elif 로직을 따로 함수화:
        bin, coincident -> pivot

        (2,2): 앞면
        (3,3): 뒷면
        (4,4): 왼쪽
        (5,5): 오른쪽        
        """
        pivot = [bin.margin_x, bin.margin_y, 0]
        # (예시) len(coincident)에 따라 여러 조건
        if len(coincident) == 4:
            pivot = [0, 0, 0]
        elif len(coincident) == 3:
            # coincident에 ('front', 'front'),('back','back),('left', 'left')가 있는 경우, 왼면이 다 붙어있는거임. 오른쪽을 띠어야하나, 다음 아이템 때 계산하면됨.
            if ('front', 'front') in coincident and ('back','back') in coincident and ('left', 'left') in coincident:
                pivot = [0, 0, 0]
            # coincident에 ('front', 'front'),('back','back'),('right', 'right')가 있는 경우, 오른면이 다 붙어있는거임. 왼쪽은 아이템만큼 띠어야함.
            elif ('front', 'front') in coincident and ('back','back') in coincident and ('right', 'right') in coincident:
                pivot = [bin.margin_x, 0, 0]
            # coincident에 ('front', 'front'),('left', 'left'),('right', 'right')가 있는 경우, 앞면이 다 붙어있는거임. 뒤쪽을 떼어야하나, 다음 아이템 때 계산하면됨.
            elif ('front', 'front') in coincident and ('left', 'left') in coincident and ('right', 'right') in coincident:
                pivot = [0,0,0]
            # coincident에 ('back','back'),('left', 'left'),('right', 'right')가 있는 경우, 뒷면이 다 붙어있는거임. 앞쪽은 아이템만큼 띠어야함.
            elif ('back','back') in coincident and ('left', 'left') in coincident and ('right', 'right') in coincident:
                pivot = [0, bin.margin_y, 0]
        elif len(coincident) == 2:
            if ('front', 'front') in coincident and ('back','back') in coincident:
                pivot = [bin.margin_x, 0, 0]
            elif ('front', 'front') in coincident and ('left', 'left') in coincident:
                pivot = [0, 0, 0]
            elif ('front', 'front') in coincident and ('right', 'right') in coincident:
                pivot = [bin.margin_x, 0, 0]
            elif ('back','back') in coincident and ('left', 'left') in coincident:
                pivot = [0, bin.margin_y, 0]
            elif ('back','back') in coincident and ('right', 'right') in coincident:
                pivot = [bin.margin_x, bin.margin_y, 0]
            elif ('left', 'left') in coincident and ('right', 'right') in coincident:
                pivot = [0, bin.margin_y, 0]
        elif len(coincident) == 1:
            if ('front', 'front') in coincident:
                pivot = [bin.margin_x, 0, 0]
            elif ('back','back') in coincident:
                pivot = [0, bin.margin_y, 0]
            elif ('left', 'left') in coincident:
                pivot = [0, bin.margin_y, 0]
            elif ('right', 'right') in coincident:
                pivot = [bin.margin_x, bin.margin_y, 0]
        else:
            pivot = [bin.margin_x, bin.margin_y, 0]

        return pivot

    def stack(self, bin, items_list):
        '''
        ZoneFit을 실행하는 메인 함수

        input: bin, items_list
        
        실행 되어야할 것:
        self.bin의 bin.items와 bin.unfit_items에 packing 결과 저장
        items는 position, rotation_quat, options가 적재된 위치의 값으로 업데이트 되어야 함
        '''

        fit_result = [-1]*len(items_list)
        self.bin = bin  # bin에 대한 정보 저장, 전역 변수로 사용
        self.items_list = items_list # item_list에 대한 정보 저장, 전역 변수로 사용

        self.set_init_bin(bin.get_all_items())  # bin에 있는 아이템을 기준으로 zone 분할

        i = 0
        fitted2 = True
        ffd = FFD_M()

        for idx, item in enumerate(items_list):
            fitted, _ = self.addItem(self.bin, item)
            # painter = PainterPlot(self.bin)
            # painter.plotZonesAndItems(self.tree.get_sorted_zones_leftover(), 
            #         title = f'result_{i}',
            #         alpha=0.2,
            #         write_num=True,
            #         fontsize=8,
            #         save=True,
            #         show = True
            #     )

            if fitted < 0:
                fitted2, loaded_item = ffd.addItem(self.bin, item)
                if fitted2:
                    # 배치된 아이템을 기준으로 self.tree 업데이트
                    intersects = self.tree.find_zones_that_intersect_item_and_corners(loaded_item)
                    for zone, corner_list in intersects:
                        # 1) 교차하는 zone을 트리에서 삭제
                        self.tree.delete(zone)

                        # 2) corner_list에서 "제일 우측상단에 있는 모서리"를 찾기
                        #    예: z > y > x 순으로 가장 큰 corner를 우측상단이라고 가정
                        top_right_corner = max(corner_list, key=lambda c: (c[2], c[1], c[0]))

                        # 3) split_zone 준비
                        #    dx, dy, dz = (top_right_corner - zone의 시작점)
                        dx = top_right_corner[0] - zone.x
                        dy = top_right_corner[1] - zone.y
                        dz = top_right_corner[2] - zone.z

                        # 4) zone을 8개 서브존으로 분할 (split_zone)
                        splitted_zones = split_zone(zone, dx, dy, dz)

                        # 5) 아이템 영역 sub-zone 은 다시 삽입하지 않음
                        for sz in splitted_zones[1:]:
                            self.tree.insert(sz)
                        merge_shared_face_zones(self.tree)
                    
                    fit_result[idx] = fitted2

            else: # fitted == True
                fit_result[idx] = fitted

            # painter = PainterPlot(self.bin)
            # painter.plotZonesAndItems(self.tree.get_sorted_zones_leftover(), 
            #         title = f'result_{i}',
            #         alpha=0.2,
            #         write_num=True,
            #         fontsize=8,
            #         save=True,
            #         show = True
            #     )
            fitted = fitted and fitted2
            
            if fitted == False and self._unfit_stop_setting:
                # bin.unfit_items의 item.name이 과 같은게 없다면 bin.unfit_items에 추가
                if not any([unfit_item.name == item.name for unfit_item in self.bin.unfit_items]):
                    self.bin.unfit_items.append(item)
                break
            elif fitted == False and not self._unfit_stop_setting:
                fit_result[idx] = False
                continue

            i += 1

        return fit_result
    
    def _xy_match(self, zone, loaded_item):
        """
        아래 아이템과 면적이 같은지 비교하는 함수
        """
        width, height, _ = loaded_item.getDimension()
        # zone의 하단에 아이템이 배치되어 있으면
        if zone.options.get('bottom_item_width'):
            if zone.options['bottom_item_width'] == width and zone.options['bottom_item_height'] == height:
                return True
        return False


    def _get_outer_leftover(self, bin, loaded_item):
        """
        bin 내에 배치된 모든 아이템에 대해,
        region1 + region2 영역 내에서 
        '실제로 아이템이 점유하는 부피'의 합을 반환.

        region1: x=0..bin.width,  y=0..50,         z=0..bin.depth
        region2: x=0..50,         y=0..bin.height, z=0..bin.depth
        """
        # 1) region1, region2의 AABB 정의
        box_r1 = (50, bin.width,
                0, 50,
                0, 50)
        box_r2 = (0, 50,
                0, bin.height,
                0, 50)

        total_overlap = 0.0

        # 2) loaded_item의 AABB 정의
        box_vertices = loaded_item.getVertices()
        box_item = (box_vertices[0][0], box_vertices[6][0],
                    box_vertices[0][1], box_vertices[6][1],
                    box_vertices[0][2], box_vertices[6][2])

        # (A) region1 교차부피
        vol_r1 = compute_overlap_volume(box_r1, box_item)

        # (B) region2 교차부피
        vol_r2 = compute_overlap_volume(box_r2, box_item)

        total_overlap += 2*vol_r1 + vol_r2

        return total_overlap
    
    def simulate_zone_placement(self,bin, zone, global_pivot, loaded_item):
        """
        'zone'에 'loaded_item'을 'margin' 위치에 배치했다고 가정하고,
        place_item_in_zone() 함수를 '가상'으로 수행한 결과를 반환.

        Returns:
            dict (tree, xy_match, max_leftover, min_leftover, etc.)
        """

        # (A) 현재 Zone 트리를 복사
        tree_copy = self.tree.copy_tree()

        # (B) zone_copy 생성
        zone_copy = copy.deepcopy(zone)

        bin_copy = copy.deepcopy(bin)

        self.place_item_in_zone(
            bin_copy,
            tree_copy,
            zone_copy,
            loaded_item,
            global_pivot
        )

        # (F) leftover 상태 계산
        max_leftover, min_leftover, avg_leftover, diag_of_max = tree_copy.get_leftover_status()

        # (G) 바깥 영역 교차 부피
        outer_lf = self._get_outer_leftover(bin_copy, loaded_item)

        # (H) xy_match
        is_xy_matched = self._xy_match(zone, loaded_item)

        # (I) vertices_after_placement
        vertices_after_placement = loaded_item.getVertices()[6]

        return {
            'tree': tree_copy,
            'xy_match': is_xy_matched,
            'max_leftover': max_leftover,
            'min_leftover': min_leftover,
            'average_leftover': avg_leftover,
            'diagonal_of_max_leftover': diag_of_max,
            'outer_leftover': outer_lf,
            'vertices_after_placement': vertices_after_placement,
        }


    def place_item_in_zone(self, bin, tree, selected_zone, item, position, sub_module_mode = False):
        """
        bin, tree, selected_zone, item, pivot 주어졌을 때:
        1) bin.store(item)
        2) selected_zone + pivot -> split
        3) tree.delete(selected_zone)
        4) sub_zones[1:] tree.insert
        5) bottom_item_* 저장
        6) merge_shared_face_zones(tree)

        주의: pivot이 '절대 좌표'인지, 'zone 내부 상대'인지에 따라 dx,dy 계산을 조정해야 함.
        여기서는 pivot이 '절대 좌표'라 가정해서,
        dx = pivot[0] - selected_zone.x + w
        ...
        """

        # 1) bin에 실제 아이템 배치
        if sub_module_mode is False:
            bin.store(item)

        # 2) split
        w,h,d = item.getDimension()
        dx = (position[0] - selected_zone.x) + w
        dy = (position[1] - selected_zone.y) + h
        dz = (position[2] - selected_zone.z) + d  # or d if zone thickness is d, etc.

        sub_zones = split_zone(selected_zone, dx, dy, dz)

        # 3) tree에서 삭제
        tree.delete(selected_zone)

        # 4) sub_zones[1:] 삽입, bottom_item_* 설정
        base_x = selected_zone.x
        base_y = selected_zone.y
        for i, sz in enumerate(sub_zones[1:], start=1):
            if sz.x == base_x and sz.y == base_y:
                sz.options['bottom_item_width'] = w
                sz.options['bottom_item_height'] = h
                sz.options['bottom_item_depth'] = d

            tree.insert(sz)

        # 5) 병합
        merge_shared_face_zones(tree)

        # 6) 필요시 leftover나 other info 반환할 수도 있음.
        return tree




