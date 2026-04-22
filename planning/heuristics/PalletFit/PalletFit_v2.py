# planning/heuristics/PalletFit.py
from planning.heuristics.base import Base
from utils.Pivot import Pivot
from utils.position import checkPivot_R
from utils.get_value import (
    get_direction_overlap,
    score_ez_distribution,
    get_score_Guillotine,
    balance_score,
)

import os, copy, sys
import concurrent.futures as cf
import multiprocessing as mp
from planning.itemManager import global_item_manager
import  os

# ───────────────────────── debug 여부 자동 감지 ─────────────────────────
def _detect_debug() -> bool:
    """
    • sys.gettrace() 가 None 이 아니면 ⇒ 어떤 디버거가 attach 된 상태
    • debugpy 가 있으면 is_client_connected() 로 한 번 더 확인 (안전)
    """
    if sys.gettrace() is not None:
        return True
    try:
        import debugpy
        return debugpy.is_client_connected()
    except ImportError:
        return False

# ────────────────────────── 가중치 · 상수 ──────────────────────────
W = {
    "ez_height_score": 22000,
    "ez_cluster_score": 7500,
    "guillotine": 25,
    "match_area_sel": 0.8,
    "match_area_all": 0.05,
    "match_rA_all": 90,
    "match_rB_all": 50,
    "weight_balance": 0.0001,   
    "bottomOverlapRatio": 1,
}

# ─────────────────── pivot 점수 + 옵션 기록 ───────────────────
def pivot_score(pivot: Pivot, loaded_item, *, bin):
    get_direction_overlap(loaded_item, bin, palletizing_mode=True)
    dg = loaded_item.options["direction_overlap"]

    guillotine = float(get_score_Guillotine(bin, loaded_item))
    bottomOvR  = float(loaded_item.getBottomOverlap())
    ez_height_score, ez_cluster_score =map(float,score_ez_distribution(bin, loaded_item))

    dirs = ("front", "left")

    # distance_sel = sum(float(dg[d][0]) for d in dirs)
    match_area_all = sum(float(dg[d][3]) for d in dg)
    match_area_sel = sum(float(dg[d][3]) for d in dirs)
    rA_sel   = sum(float(dg[d][1]) for d in dirs)
    rB_sel   = sum(float(dg[d][2]) for d in dirs)
    w_balance = float(balance_score(bin, cand_item=loaded_item))

    # 디버깅용 세부 점수 기록
    pivot.options = {
        "ez_height_score": (ez_height_score, W["ez_height_score"] * ez_height_score),
        "ez_cluster_score": (ez_cluster_score, W["ez_cluster_score"] * ez_cluster_score),
        "guillotine":     (guillotine,     W["guillotine"]         * guillotine),
        # "distance_sel":   (distance_sel,   W["match_area_sel"]           * distance_sel),
        "match_area_all":  (match_area_all,       W["match_area_all"]      * match_area_all),
        "match_area_sel":       (match_area_sel,       W["match_area_sel"]           * match_area_sel),
        "rA_sel":         (rA_sel,         W["match_rA_all"]       * rA_sel),
        "rB_sel":         (rB_sel,         W["match_rB_all"]       * rB_sel),
        "w_balance":      (w_balance,      W["weight_balance"]     * w_balance),
        "bottomOvR":      (bottomOvR,      W["bottomOverlapRatio"] * bottomOvR),
        "dirs": dirs,
    }

    return (
        W["ez_height_score"] * ez_height_score +
        W["ez_cluster_score"] * ez_cluster_score +
        W["guillotine"]     * guillotine +
        W["match_area_all"]  * match_area_all   +
        W["match_area_sel"]       * match_area_sel   +
        W["match_rA_all"]   * rA_sel     +
        W["match_rB_all"]   * rB_sel     +
        W["weight_balance"] * w_balance  +
        W["bottomOverlapRatio"] * bottomOvR
    )
# ────────────────────────── 워커용 함수 ──────────────────────────
# 워커용 함수  (pivot 1개당 평가)
def _eval_pivot(args):
    (px, py, pz, rt, direction,
     bin_clone, item, items_snapshot) = args

    # 0) 레지스트리 재구성  ──────────────────────────────
    from planning.itemManager import global_item_manager as gm
    gm._id_to_item = {it._id: it for it in items_snapshot}
    gm._next_id    = max(gm._id_to_item, default=-1) + 1   # ★ 수정

    # 1) 피벗 적합성 체크
    fitted, loaded_item = checkPivot_R(bin_clone, item,
                                       [px, py, pz], rt)
    if fitted < 0:
        return None

    pv    = Pivot(px, py, pz, rt, direction=direction,
                  bench_bin=bin_clone)
    score = pivot_score(pv, loaded_item, bin=bin_clone)
    return score, (px, py, pz, rt, direction), pv.options

# ─────────────────────────── 클래스 ───────────────────────────
class PalletFit(Base):
    DEBUG_MODE =  _detect_debug()      # ← True: 디버깅, False: 일반 실행
    # DEBUG_MODE = True  # 디버깅 모드 비활성화 (기본값)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
    def stack(self, bin, items_list):
        fit_result = [-1] * len(items_list)
        index_map  = {it._id: i for i, it in enumerate(items_list)}

        if bin.corner != 0 and bin.size == 0:
            for i, cube in enumerate(self.addCorner(bin)):
                self.putCorner(bin, i, cube)

        items_list.sort(key=lambda it: it.depth,     reverse=True)
        items_list.sort(key=lambda it: it.loadbear,  reverse=True)

        for item in items_list:
            fitted, _ = self.addItem(bin, item)
            self._mark_fit_result(item, fitted, fit_result, index_map)

        return fit_result

    # ---------------------- addItem ----------------------
    def addItem(self, bin, item, *, chunksize=64):
        self.set_init_PT(bin, bin.get_all_items())

        # 1) 후보 pivot 수집
        checklist_pivots, seen = [], set()
        def _add_if_new(pv, tol=2):
            key = (round(pv.x,tol), round(pv.y,tol), round(pv.z,tol),
                   tuple(round(v,tol) for v in pv.rt))
            if key not in seen:
                seen.add(key); checklist_pivots.append(pv)

        pivot_list = list(bin.pivotTree.in_order_traversal()) 

        for pivot in pivot_list:
            pivot.bench_bin = bin
            _add_if_new(pivot)

            # vx, vy, vz = pivot.x, pivot.y, pivot.z
            # w, h, _    = item.getDimension()
            # pair_rt = RotationType.get_rotation_pair(pivot.rt)
            # dir_, corner = pivot.direction.split('-')[0], pivot.direction.split('-')[-1]
            # if corner == '7':
            #     _add_if_new(Pivot(vx,   vy-h, vz, pivot.rt, direction=pivot.direction, bench_bin=bin))
            #     _add_if_new(Pivot(vx,   vy-w, vz, pair_rt, direction=pivot.direction, bench_bin=bin))
            # elif corner == '5':
            #     _add_if_new(Pivot(vx-w, vy,   vz, pivot.rt, direction=pivot.direction, bench_bin=bin))
            #     _add_if_new(Pivot(vx-h, vy,   vz, pair_rt, direction=pivot.direction, bench_bin=bin))
            # if dir_ in ('down','up') and corner == '6':
            #     _add_if_new(Pivot(vx-w, vy-h, vz, pivot.rt, direction=pivot.direction, bench_bin=bin))
            #     _add_if_new(Pivot(vx-h, vy-w, vz, pair_rt, direction=pivot.direction, bench_bin=bin))

        if not checklist_pivots:
            return False, None
        

        # ── ❶  평가  ──────────────────────────────────────────────
        if self.DEBUG_MODE:  # 단일 프로세스 평가
            # 2) 평가 (순차 or 멀티프로세스)
            feasible = []

            for pv in checklist_pivots:
                fitted, loaded_item = checkPivot_R(bin, item, [pv.x, pv.y, pv.z], pv.rt)
                if fitted < 0:
                    pv.options['fitted'] = fitted
                    continue
                sc = pivot_score(pv, loaded_item, bin=bin)
                feasible.append((sc,
                                    (pv.x,pv.y,pv.z,pv.rt,pv.direction),
                                    pv.options))

            if not feasible:
                return False, None

        else:   # 병렬 프로세스 평가
            # 2) 워커에게 전달할 공용 스냅샷 ------------------------------------------------------
            bin_clone      = copy.deepcopy(bin)                # 전체 bin 복사
            items_snapshot = [copy.deepcopy(global_item_manager.get(i))
                            for i in bin.item_ids]           # id → Item 복사본

            tuple_args = [
                (pv.x, pv.y, pv.z, pv.rt, pv.direction,
                bin_clone, item, items_snapshot)
                for pv in checklist_pivots
            ]

            # 3) 병렬 평가 -----------------------------------------------------------------------
            feasible = []
            n_worker = max(os.cpu_count() - 2, 1)
            ctx      = mp.get_context("fork" if os.name != "nt" else "spawn")

            with cf.ProcessPoolExecutor(max_workers=n_worker,
                                        mp_context=ctx) as pool:
                for res in pool.map(_eval_pivot, tuple_args, chunksize=chunksize):
                    if res:
                        feasible.append(res)

            if not feasible:
                return False, None
            
        # 4) 최고 점수 선택 -------------------------------------------------------------------
        best_score, best_pv, best_opts = max(feasible, key=lambda t: (t[0], -t[1][2], t[1][0], t[1][1]))
        px, py, pz, rt, direction = best_pv
        best_pivot = Pivot(px, py, pz, rt, direction=direction, bench_bin=bin)
        best_pivot.options = best_opts

        _, placed_item = checkPivot_R(bin, item, [px, py, pz], rt)

        bin.store(placed_item) 
        self.store2Pivot(bin)

        return True, placed_item