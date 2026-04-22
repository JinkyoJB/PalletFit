# planning/heuristics/base.py
from utils.Pivot import PivotAVLTree
from utils.Pivot import Pivot
from planning.itemManager import global_item_manager
from planning.item import Item, RotationType

from utils.pivot_generation import (
    project_lines_left_to_pivots,
    project_lines_front_to_pivots,
    project_lines_down_to_pivots,
    project_lines_down_to_pivots2left,
    project_lines_down_to_pivots2front,
    get_pivots_ep,
    get_pivots_cp,
    get_pivots_ems,
    merge_close_pivots,
    
) 
from utils.painter import PainterPlot
from collections import defaultdict
# from dataclasses import asdict

class Base:
    def __init__(self, unfit_stop_setting = True, rotation_type = "BasicRotation"):
        self._unfit_stop_setting = unfit_stop_setting
        self._rotation_type = rotation_type
        self.unfit_items = set()  # 적재 실패한 아이템들

    def __repr__(self):
        return self.__class__.__name__
    
    def set_init_PT(self, bin, item_list):
        '''
        bin에 기존에 있던 아이템들을 pivotTree에 추가
        '''
        bin.pivotTree = PivotAVLTree()
        if bin.origin_item_id is None:
            bin.pivotTree.insert(Pivot( 0, 0, 0, RotationType.RT_WHD, bench_bin=bin))
            bin.pivotTree.insert(Pivot( 0, 0, 0, RotationType.RT_HWD, bench_bin=bin))
        else:
            for _ in item_list:
                self.store2Pivot(bin)

    def _render_pivot_set(self,
                        bin_obj,
                        tag: str,
                        pivot_generators,
                        show=True):
        """
        • pivot_generators : [(이름, pivot 리스트), ...]
        • 한 세트(그림)마다 pivotTree 를 새로 만들어 insert 후 렌더
        """
        bin.pivotTree = PivotAVLTree()
        for name, pivots in pivot_generators:
            for pv in pivots:
                bin.pivotTree.insert(pv)

        if show:
            PainterPlot(bin_obj).plotBoxAndItems(
                title=f'pivotTree_{tag}_{bin_obj.size}_{bin.pivotTree.size}',
                alpha=0.25,
                save=True, show=True,
                save_path='planning/renders/debug/Pivot/',
                view_azim=87,
                view_elev=1,
                view_roll=None,
                view_dist=1,
                pivot_mark_size=10,
                save_dpi=800,  # 해상도 높이기
            )


    def store2Pivot(self, bin):
        """
        • 기존 pivot 좌표에 존재하는 모든 회전 쿼터니언 삭제
        • 방향 순서대로 pivot 생성 → 각 리스트를 merge_close_pivots로 병합 → 트리에 삽입
        • 방향별로 시각화 훅이 있으면 즉시 갱신
        """
        # ── 0) 기존 pivotTree 초기화 ───────────────
        bin.pivotTree = PivotAVLTree()

        # ── 1) 방향별 pivot 생성 ────────────────────────────────────
        left_pivots   = project_lines_left_to_pivots(bin)
        front_pivots  = project_lines_front_to_pivots(bin)
        down_pivots   = project_lines_down_to_pivots(bin)
        left2_pivots  = project_lines_down_to_pivots2left(bin,  down_pivots)
        front2_pivots = project_lines_down_to_pivots2front(bin, down_pivots)
        # cp_pivots   = get_pivots_cp(bin)
        # ep_pivots   = get_pivots_ep(bin)
        # ems_pivots  = get_pivots_ems(bin)

        dir_functions = [
            ("left",   left_pivots),
            ("front",  front_pivots),
            ("down",   down_pivots),
            ("left2",  left2_pivots),
            ("front2", front2_pivots),
            
        ]

        # ── 2) 병합 파라미터 ───────────────────────────────────────
        tol_mm = max(bin.margin_x, bin.margin_y)
        keep   = "first" # 'first' or 'avg'

        all_merged = merge_close_pivots(
            [pv for _, lst in dir_functions for pv in lst], tol_mm=tol_mm, keep=keep
        )
        for pv in all_merged:
            bin.pivotTree.insert(pv)
        return len(all_merged)


    def stack(self, bin, items_list):
        """
        bin에 items_list를 적재하되, composite(부모) 아이템도 생성될 수 있음.
        items_list는 원본 아이템들.
        """

        # 1) 원본 아이템의 ID 기준으로 index_map 생성
        index_map = {it._id: i for i, it in enumerate(items_list)}

        # 2) fit_result 초기화
        fit_result = [-1] * len(items_list)

        # 코너에 보호용 큐브 추가
        if bin.corner != 0 and bin.size == 0:
            corner_list = self.addCorner(bin)
            for i in range(len(corner_list)):
                self.putCorner(bin, i, corner_list[i])

        # 3) Composite 구성
        grouped_items_list = self.group_items_with_top(bin, items_list)
        grouped_items_list.sort(key=lambda x: (not x.is_composite, -(x.width * x.height)))

        # 4) 아이템 적재
        for comp_item in grouped_items_list:
            fitted, _ = self.addItem(bin, comp_item)
            self._mark_fit_result(comp_item, fitted, fit_result, index_map)

        # 5) 실패한 remainder 재도전
        if fit_result.count(1) != len(items_list):
            remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            for item in remainder_items:
                fitted, _ = self.addItem(bin, item)
                if fitted > 0:
                    print(f'{item.name}--------------remainder--------------')
                    self._mark_fit_result(item, fitted, fit_result, index_map)

        if fit_result.count(1) != len(items_list):
            # 그래도 실패 -> surface_ratio 0.7로 하여 호출
            bk_surface_ratio = bin.support_surface_ratio
            bin.support_surface_ratio = 0.7
            remainder_items = [items_list[i] for i, res in enumerate(fit_result) if res == 0]
            if fit_result.count(1) != len(items_list):
                for item in remainder_items:
                    fitted, _ = self.addItem(bin, item)
                    if fitted > 0:
                        print(f'{item.name}--------------bk_surface_ratio1--------------')
                        self._mark_fit_result(item, fitted, fit_result, index_map)
            bin.support_surface_ratio = bk_surface_ratio


        # 실패한 리스트 self.unfit_items에 추가
        for i, res in enumerate(fit_result):
            if res == 0:
                self.unfit_items.add(items_list[i])

        return fit_result
    
    def _mark_fit_result(self, parent_item, fitted, fit_result, index_map):
        """
        parent_item과 그 자식들(children_ids)에 대해 성공/실패 결과 기록.
        """
        # A) parent_item 자체 기록
        if parent_item._id in index_map:
            idx = index_map[parent_item._id]
            fit_result[idx] = 1 if fitted else 0

        # B) 재귀적으로 자식 기록
        for child_id in parent_item.children_ids:
            child = global_item_manager.get(child_id)
            if child:
                self._mark_fit_result(child, fitted, fit_result, index_map)

        
    def addCorner(self, bin):
        '''
        bin에 corner 추가
        '''
        if bin.corner != 0:
            corner =bin.corner
            corner_list = [Item(
                    partno = f'corner{i}',
                    name = f'corner{i}',
                    objshape = 'cube',
                    width = corner,
                    height = corner,
                    depth = corner,
                    rotation_quat= [0.0, 0.0, 0.0, 1.0],
                    priority = 0,
                    updown = False,
                    weight = 0,
                    loadbear = 0,
                    unit = bin.unit
                )
                for i in range(8)]

            return corner_list
        
    def putCorner(self, bin, idx, corner):
        '''
        corner를 bin에 적재
        '''
        x = bin.width - corner.width
        y = bin.height - corner.height
        z = bin.depth - corner.depth

        pos = [[0, 0, 0], [0, 0, z], [0, y, z], [0, y, 0],
            [x, y, 0], [x, 0, 0], [x, 0, z], [x, y, z]]
        corner.b_position = pos[idx]
        bin.items.append(corner)


    def group_items_with_top(self, bin, items_list):
        """
        [개요]
        1) bin.itemTree에서 'top items' (위에 다른 아이템이 없는 아이템)을 찾는다.
        - 각 top item은 (W, H)를 기준으로 그룹을 만든다. 
            (이때, 해당 top item 자체를 stack에 넣진 않음 - 이미 bin에 있는 아이템이므로 중복 적재를 피하기 위함)
        - top item이 점유한 높이만큼 bin.depth에서 차감하여 remain_depth를 계산한다.
            => 이 remain_depth를 바탕으로 새로 들어오는 items_list 중 (W,H)가 동일한 아이템을 해당 그룹에 적재할 수 있는지 결정.

        2) items_list의 아이템들도 (W,H) 기준으로 분류 후,
        - 만약 기존 wh_groups[(W,H)] 중에서 remain_depth가 충분히 남아 있는 그룹이 있으면 그 그룹에 쌓는다.
        - 없다면 새 그룹을 만든다(왜냐하면 동일 (W,H)라도 top 아이템이 달라 다른 높이 제한이 있을 수 있기 때문).

        3) 최종적으로 각 그룹에서 stack한 아이템들이
        - 1개뿐이면 그대로 반환,
        - 2개 이상이면 Composite Item으로 만들어서 반환한다.

        [반환값]
        - new_items_list: 위 과정을 통해 새로운 아이템(Composite 포함)이 생겼다면 그 목록을 반환
        """

        wh_groups = defaultdict(list)

        for top_it in bin.get_visible_items_topdown():
            tw, th, td = top_it.getDimension()

            # 남은 높이 = bin.depth - (top_it가 이미 차지한 z + d)
            remain_depth = bin.depth - top_it.ez
            if remain_depth < 0:
                remain_depth = 0

            # W, H 중 더 작은 걸 앞에 두는 사용자 규칙 (ex. W <= H 형태로)
            W, H = (tw, th) if tw > th else (th, tw)

            # 이미 존재하는 (W,H) 그룹에 'remain_depth'가 같은 top이 있을 수도 있으니
            # 그건 상황에 따라 병합/생성하는 로직을 넣을 수 있지만,
            # 여기서는 단순화하여 각각 새 group_info를 추가한다.
            group_info = {
                'stack': [],          # top_it을 다시 적재하지 않음
                'remain_depth': remain_depth,
                'base_w': W,
                'base_h': H,
            }
            wh_groups[(W, H)].append(group_info)

        # ---------------------------------
        # 3) items_list를 (W,H)별로 넣을 그룹을 찾기
        # ---------------------------------
        for item in items_list:
            W, H, D = item.getDimension()

            # (W<H)이면 rotation 토글
            if W < H:
                r_pair = RotationType.get_rotation_pair(item.rotation_quat)
                if r_pair is not None:
                    item.rotation_quat = r_pair
                    W, H, D = item.getDimension()

            # 다시 한번 (W <= H) 형태로 맞춤
            key = (W, H)

            # (W,H) 그룹이 아직 하나도 없다면 => 새로 만든다
            if key not in wh_groups:
                new_group_info = {
                    'stack': [],
                    'remain_depth': bin.depth,
                    'base_w': W,
                    'base_h': H,
                }
                wh_groups[key].append(new_group_info)

            # 사용할 수 있는 group_info가 있는지 탐색
            # (remain_depth >= D)를 만족하는 그룹이 있어야 넣을 수 있음
            chosen_group = None
            for gi in wh_groups[key]:
                if gi['remain_depth'] >= D:
                    chosen_group = gi
                    break

            # 만약 적당한 group이 없다면 => 새 group_info를 추가
            if chosen_group is None:
                chosen_group = {
                    'stack': [],
                    'remain_depth': bin.depth,
                    'base_w': W,
                    'base_h': H,
                }
                wh_groups[key].append(chosen_group)

            # chosen_group에 item을 적재
            chosen_group['stack'].append(item)
            chosen_group['remain_depth'] -= D  # 사용한 높이만큼 남은 높이를 줄임

        # ---------------------------------
        # 4) 최종 정리:
        #    stack에 아이템이 2개 이상이면 Composite로 묶고,
        #    1개뿐이면 그대로 반환
        # ---------------------------------
        new_items_list = []
        for (W, H), group_info_list in wh_groups.items():
            for gi in group_info_list:
                st = gi['stack']
                if not st:
                    # 아무 아이템도 쌓이지 않은 그룹은 스킵
                    continue
                if len(st) == 1:
                    # 1개뿐이라면 그대로 사용
                    new_items_list.append(st[0])
                else:
                    # 여러 아이템이 쌓였다면 Composite로 묶기
                    total_depth = sum(x.getDimension()[2] for x in st)
                    comp = bin.create_composite_item(
                        children=st,
                        base_w=W,
                        base_h=H,
                        stacked_d=total_depth
                    )
                    new_items_list.append(comp)

        return new_items_list

