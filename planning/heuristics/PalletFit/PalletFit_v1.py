# planning/heuristics/PalletFit.py
from planning.item import RotationType
from planning.heuristics.base import Base
from utils.Pivot import Pivot

from utils.position import checkPivot_R
from utils.get_value import get_direction_overlap, flat_surface_weight, distance_sq_from_origin, get_score_Guillotine, weight_balance_score

import time
# ── 하부층: flat-surface·접촉길이·지지면을 특히 중시
LOWER_W = {
    "flat"          : 100,
    "guillotine"    : 500,
    "match_len_all" : 2,
    "match_len_f"   : 2,
    "match_len_l"   : 1,
    "ratioA_f"      : 100,
    "ratioA_l"      : 50,
    "ratioB_f"      : 100,
    "ratioB_l"      : 50,
    "pos"           : -1,   # ← 작을수록 유리 → 음의 weight
    # "dist"          : -1e-3,   # 원점(0,0) 가까울수록 +
    "weight_balance": -1000,   # 무게 균형
    "bottomOverlapRatio" : 0.1,   # 접촉면적
}

CHANGE_DEPTH = 0.6   # 깊이 변경 기준 (0.5m) — 하부/상부 구분용

# ── 상부층: XY-coverage & 중앙 배치 쪽에 더 weight
UPPER_W = {
    "flat"          : 100,
    "guillotine"    : 500,
    "match_len_all" : 2,
    "match_len_f"   : 2,
    "match_len_l"   : 1,
    "ratioA_f"      : 100,
    "ratioA_l"      : 50,
    "ratioB_f"      : 100,
    "ratioB_l"      : 50,
    "pos"           : -1,   # ← 작을수록 유리 → 음의 weight
    # "dist"          : -1e-3,   # 원점(0,0) 가까울수록 +
    "weight_balance": -2000,   # 무게 균형
    "bottomOverlapRatio" : 0.1,   # 접촉면적

}

class PalletFit(Base):
    '''
    1.1 * 1.1 *1.8 m의 팔레트에 안정적으로 적재하는 알고리즘
    xy면에서 아이템은 x 또는 y 축으로 한축은 무조건 bin과 접하는 면이 가득 차야한다. -> 그래야 지지축이 생김
    지지축은 엇갈려 있는게 베스트. (ex: 1: x축, 2: y축, 3: x축, 4: y축, …)

    주어지는 아이템의 개수는 아이템의 윗면적의 합이 팔레트의 면적의 1.5배 이상이 되도록 한다. (안정적인 조합 확보)
    '''
    def __init__(self, unfit_stop_setting=True, rotation_type=RotationType.BasicRotation):
        super().__init__(unfit_stop_setting, rotation_type)

    def stack(self, bin, items_list):
        fit_result = [-1] * len(items_list)
        index_map = {it._id: i for i, it in enumerate(items_list)}

        # 1) 코너 큐브
        if bin.corner != 0 and bin.size == 0:
            corner_list = self.addCorner(bin)
            for i in range(len(corner_list)):
                self.putCorner(bin, i, corner_list[i])
        
        # it.loadbear이 높은 순으로 정렬
        items_list.sort(key=lambda it: it.loadbear, reverse=True)
        # it.depth가 높은 순으로 정렬
        items_list.sort(key=lambda it: it.depth, reverse=True)

        # 4) 아이템 적재
        for item in items_list:
            fitted, _ = self.addItem(bin, item)
            # if fitted > 0:        # RL eval용
            #     time_stamp = f"{time.time():.3f}_{item._id}"
            #     bin.render(
            #  write_num=True, name=time_stamp,
            #         save = True,
            #         show= False,
            #         save_path="planning/RL/SB3/eval_snaps/100steps",
            #     )

            self._mark_fit_result(item, fitted, fit_result, index_map)
            if fitted > 0:
                bin.post_merge(item._id)  # 아이템 적재 후, bin에 merge


        return fit_result
    
    def addItem(self, bin, item):
        """
        1) pivotTree가 비어 있으면, 첫 Pivot(예: 0,0,0) 삽입
        2) pivotTree에 있는 모든 Pivot을 순회하며:
            - checkPivot_R로 배치 가능 여부 확인
            - 가능하면 (pivot, loaded_item) 쌍을 feasible_pivots에 저장
        3) feasible_pivots가 비어 있으면 배치 실패
        4) 동적 지표를 기준으로 정렬하여 최적 Pivot을 선택
        5) 실제 bin.store(...) 후, self.store2Pivot(...)로 처리
        """
        # 3) 동적 지표에 따른 정렬:
        def pivot_score(pivot, loaded_item, *, bin):
            # 시각화
            # bin.render( write_num=True, name=fitted, pivots=[[pivot.x, pivot.y, pivot.z], [loaded_item.ex, loaded_item.ey, loaded_item.ez]])
            """가중치 합산 방식 점수 — 클수록 좋은 pivot (float 반환)"""
            get_direction_overlap(loaded_item, bin)
            dg = loaded_item.options['direction_overlap']

            mlen_F  = float(dg['front'][3])
            mlen_L  = float(dg['left'][3])
            mlenAll = mlen_F + mlen_L + float(dg['right'][3]) + float(dg['back'][3])

            rA_F, rA_L = map(float, (dg['front'][1], dg['left'][1]))
            rB_F, rB_L = map(float, (dg['front'][2], dg['left'][2]))

            guillotine = float(get_score_Guillotine(bin, loaded_item)["total"])
            flat_w     = float(flat_surface_weight(bin, loaded_item))

            w_balance = float(weight_balance_score(bin, loaded_item))
            bottomOverlapRatio = float(loaded_item.getBottomOverlap())

            depth_half = CHANGE_DEPTH * float(bin.depth)
            if pivot.z <= depth_half:                        # 하부
                pos = float(min(pivot.x, bin.width-pivot.x,
                                pivot.y, bin.height-pivot.y))
                W = self.LOWER_W
            else:                                            # 상부
                cx, cy = bin.width*0.5, bin.height*0.5
                pos = float(abs(pivot.x-cx) + abs(pivot.y-cy))
                W = self.UPPER_W

            # dist = float(distance_sq_from_origin(loaded_item))
            pivot.options = {
                'flat_w': (flat_w,W["flat"]           * flat_w ),
                'guillotine': (guillotine, W["guillotine"] * guillotine),
                'mlenAll': (mlenAll, W["match_len_all"] * mlenAll),
                'mlen_F': (mlen_F, W["match_len_f"] * mlen_F),
                'mlen_L': (mlen_L, W["match_len_l"] * mlen_L),
                'rA_F': (rA_F, W["ratioA_f"] * rA_F),
                'rA_L': (rA_L, W["ratioA_l"] * rA_L),
                'rB_F': (rB_F, W["ratioB_f"] * rB_F),
                'rB_L': (rB_L, W["ratioB_l"] * rB_L),
                'pos': (pos, W["pos"] * pos),
                'weight_balance': (w_balance, W["weight_balance"] * w_balance),
                'bottomOverlapRatio': (bottomOverlapRatio, W["bottomOverlapRatio"] * bottomOverlapRatio),
            }
            score = (
                W["flat"]           * flat_w      +
                W["guillotine"]     * guillotine  +
                W["match_len_all"]  * mlenAll     +
                W["match_len_f"]    * mlen_F      +
                W["match_len_l"]    * mlen_L      +
                W["ratioA_f"]       * rA_F        +
                W["ratioA_l"]       * rA_L        +
                W["ratioB_f"]       * rB_F        +
                W["ratioB_l"]       * rB_L        +
                W["pos"]            * pos         +
                W["weight_balance"] * w_balance + 
                W["bottomOverlapRatio"]  * bottomOverlapRatio
            )
            return float(score)           # ★ 반드시 float!

        
        self.set_init_PT(bin, bin.get_all_items())

        # 2) 배치 가능한 pivot 후보들 수집
        checklist_pivots: list[Pivot] = []
        seen: set[tuple] = set()          # 이미 추가된 pivot key 저장

        def _add_if_new(pv: Pivot, *, tol: int = 2):
            """중복이면 건너뛰고, 새 pivot이면 리스트/집합에 등록"""
            key = (round(pv.x, tol),
                round(pv.y, tol),
                round(pv.z, tol),
                tuple(round(v, tol) for v in pv.rt))
            if key not in seen:
                seen.add(key)
                checklist_pivots.append(pv)

        # 1) 기존 트리의 pivot 들
        for pivot in bin.pivotTree.in_order_traversal():
            pivot.bench_bin = bin
            _add_if_new(pivot)            # 원본 pivot

            # 2) 방향별 “보정 pivot”도 추가
            vx, vy, vz   = pivot.x, pivot.y, pivot.z
            w,  h,  d    = item.getDimension()
            dir = pivot.direction.split('-')[0]  # 방향 정보
            corner = pivot.direction.split('-')[-1]

            if corner in ('3', '7'):
                _add_if_new(Pivot(vx,     vy - h, vz, pivot.rt,
                                direction=pivot.direction, bench_bin=bin))
            elif corner == '5':
                _add_if_new(Pivot(vx - w, vy,     vz, pivot.rt,
                                direction=pivot.direction, bench_bin=bin))
            if dir in ('down', "up") and corner in ('2', '6'):
                _add_if_new(Pivot(vx - w, vy - h, vz, pivot.rt,
                                direction=pivot.direction, bench_bin=bin))
            elif dir == 'front' and corner == "2":
                _add_if_new(Pivot(vx,     vy - h, vz, pivot.rt,
                                direction=pivot.direction, bench_bin=bin))
                
            # bin.render( write_num=True,name=fitted, pivots=[[pivot.x, pivot.y, pivot.z], [loaded_item.ex, loaded_item.ey, loaded_item.ez]])
            

        if not checklist_pivots:
            # 후보가 전혀 없다면 실패
            return False, None
        
        feasible_pivots = []
        for pv in checklist_pivots:
            fitted, loaded_item = checkPivot_R(bin, item,
                                            [pv.x, pv.y, pv.z], pv.rt)
            if fitted > 0:
                sc = pivot_score(pv, loaded_item, bin=bin)
                feasible_pivots.append((pv, loaded_item, sc))
        # 시각화
        # bin.render( write_num=True, name=fitted, pivots=[[pivot.x, pivot.y, pivot.z], [loaded_item.ex, loaded_item.ey, loaded_item.ez]])
        if not feasible_pivots:
            # 배치 가능한 pivot이 없다면 실패
            return False, None

        # “점수 큰 → 좋은”  정렬 (내림차순)
        feasible_pivots.sort(key=lambda t: t[2], reverse=True)
        best_pivot, best_loaded_item, _ = feasible_pivots[0]
        # bin.render( write_num=True, name=fitted, pivots=[[best_pivot.x, best_pivot.y, best_pivot.z], [best_loaded_item.ex, best_loaded_item.ey, best_loaded_item.ez]])

        # 4) bin에 최종 저장 + pivot 관련 처리
        bin.store(best_loaded_item)
        self.store2Pivot(bin)

        # bin.render(write_num=True)
        return True, best_loaded_item

        





