# planning/RL/PalletFit_RL/act_builder.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from utils.pivot_generation import (
    project_lines_left_to_pivots,
    project_lines_front_to_pivots,
    project_lines_down_to_pivots,
    project_lines_down_to_pivots2left,
    project_lines_down_to_pivots2front,
    project_vertices_top_to_pivots,
    merge_close_pivots,
    get_pivots_ep,
    get_pivots_cp,
    get_pivots_ems,
)
from utils.checkPivot import checkPivot_R, SUCCESS
from utils.util_functions import clip01, eps
from planning.item import RotationType
from utils.Pivot import Pivot, PivotAVLTree

from planning.RL.PalletFit_RL.config import (
    ACTION_MAX_CANDIDATES,
    PIVOT_FEAT_DIM,
    ALLOWED_ROT_IDS,
    DEDUP_ROUND_NDIGITS,
)
from planning.RL.PalletFit_RL.obs_builder import queue_head, get_look_items
from planning.itemManager import global_item_manager

# ─────────────────────────────────────────────────────────────
# Pivot 후보 생성 (front / left) + 중복 제거 (기존 유지)
# ─────────────────────────────────────────────────────────────
def get_pivots_for_dir(dir_name: str, bin_obj):
    rd = DEDUP_ROUND_NDIGITS

    def _dedup_keep_order(seq):
        uniq, seen = [], set()
        for pv in seq:
            rid = RotationType.quat_to_index(getattr(pv, "rt", None))
            if rid is None or (ALLOWED_ROT_IDS and rid not in ALLOWED_ROT_IDS):
                continue
            key = (
                round(float(pv.x), rd),
                round(float(pv.y), rd),
                round(float(pv.z), rd),
                int(rid),
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append(pv)
        return uniq

    if getattr(bin_obj, "origin_item_id", None) is None:
        piv0 = []
        for rid in ALLOWED_ROT_IDS or [0, 1]:
            rt = RotationType.index_to_quat(rid)
            if rt is not None:
                piv0.append(Pivot(0, 0, 0, rt, bench_bin=bin_obj))
        return piv0

    if dir_name == "front":
        lst = []
        lst.extend(project_lines_front_to_pivots(bin_obj))                 
        cands_d = project_lines_down_to_pivots(bin_obj)                    
        lst.extend(cands_d)                                                
        lst.extend(project_lines_down_to_pivots2front(bin_obj, cands_d))   
        return _dedup_keep_order(lst)

    elif dir_name == "left":
        lst = []
        lst.extend(project_lines_left_to_pivots(bin_obj))                  
        cands_d = project_lines_down_to_pivots(bin_obj)                    
        lst.extend(cands_d)                                                
        lst.extend(project_lines_down_to_pivots2left(bin_obj, cands_d))    
        return _dedup_keep_order(lst)
    
    elif dir_name == "up":
        lst = []
        lst.extend(project_vertices_top_to_pivots(bin_obj))                
        return _dedup_keep_order(lst)

    return []


def Edge_Projection(bin) -> List[Pivot]:
    if getattr(bin, "origin_item_id", None) is None:
        piv0 = []
        for rid in (ALLOWED_ROT_IDS or [0, 1]):
            rt = RotationType.index_to_quat(rid)
            if rt is not None:
                piv0.append(Pivot(0, 0, 0, rt, bench_bin=bin))
        return piv0

    left_pivots   = project_lines_left_to_pivots(bin)
    front_pivots  = project_lines_front_to_pivots(bin)
    down_pivots   = project_lines_down_to_pivots(bin)
    left2_pivots  = project_lines_down_to_pivots2left(bin,  down_pivots)
    front2_pivots = project_lines_down_to_pivots2front(bin, down_pivots)

    dir_functions = [
        ("left",   left_pivots),
        ("front",  front_pivots),
        ("down",   down_pivots),
        ("left2",  left2_pivots),
        ("front2", front2_pivots),
    ]

    tol_mm = max(bin.margin_x, bin.margin_y) if hasattr(bin, 'margin_x') and hasattr(bin, 'margin_y') else 5
    keep   = "first" 

    all_merged = merge_close_pivots(
        [pv for _, lst in dir_functions for pv in lst], tol_mm=tol_mm, keep=keep
    )
    return all_merged

# ─────────────────────────────────────────────────────────────
# Pivot <-> tensor encoding (기존 유지)
# ─────────────────────────────────────────────────────────────
def encode_pivot(pv: Pivot, bin_obj) -> np.ndarray:
    rid = RotationType.quat_to_index(getattr(pv, "rt", None))
    if rid is None or (ALLOWED_ROT_IDS and rid not in ALLOWED_ROT_IDS):
        raise ValueError(f"Invalid pivot rotation: {pv.rt}")

    return np.array([
        clip01(pv.x / eps(bin_obj.width)),
        clip01(pv.y / eps(bin_obj.depth)),
        clip01(pv.z / eps(bin_obj.height)),
        float(rid)
    ], dtype=np.float32)


def decode_pivot(arr: np.ndarray, bin_obj) -> Pivot:
    if len(arr) != PIVOT_FEAT_DIM:
        raise ValueError(f"Invalid array length for pivot decoding: {len(arr)}")
    x = float(arr[0]) * eps(bin_obj.width)
    y = float(arr[1]) * eps(bin_obj.depth)
    z = float(arr[2]) * eps(bin_obj.height)
    rid = int(arr[3])
    rt = RotationType.index_to_quat(rid)
    if rt is None or (ALLOWED_ROT_IDS and rid not in ALLOWED_ROT_IDS):
        raise ValueError(f"Invalid rotation id for pivot decoding: {rid}")

    return Pivot(x, y, z, rt, bench_bin=bin_obj)

def encode_action(slot: int, k: int, K: int) -> int:
    return slot * K + k

def decode_action(action: int, K: int) -> tuple[int, int]:
    slot = int(action // K)
    k    = int(action % K)
    return slot, k

# ─────────────────────────────────────────────────────────────
# Action mask / candidates (단일 아이템용 - 기존 유지)
# ─────────────────────────────────────────────────────────────
def get_action_mask(bin_obj, current_item, cands_option='EDP') -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((ACTION_MAX_CANDIDATES,), dtype=bool)
    cands_arr = np.zeros((ACTION_MAX_CANDIDATES, PIVOT_FEAT_DIM), dtype=np.float32)
    feasible: list[Pivot] = []

    if cands_option == 'CP':
        candidates = get_pivots_cp(bin_obj)
    elif cands_option == 'EP':
        candidates = get_pivots_ep(bin_obj)
    elif cands_option == 'EMS':
        candidates = get_pivots_ems(bin_obj)
    else:  # 'EDP'
        candidates = Edge_Projection(bin_obj)

    for pv in candidates:
        code, _ = checkPivot_R(bin_obj, current_item, [pv.x, pv.y, pv.z], pv.rt, apply_margin=True)
        if code == SUCCESS:
            feasible.append(pv)

    limit = min(len(feasible), ACTION_MAX_CANDIDATES)
    bin_obj.pivotTree = PivotAVLTree() 

    for i, pv in enumerate(feasible[:limit]):
        mask[i] = True
        bin_obj.pivotTree.insert(pv)
        cands_arr[i] = encode_pivot(pv, bin_obj)

    return mask, cands_arr

def get_action_mask_wo(bin_obj, cands_option='EDP') -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((ACTION_MAX_CANDIDATES,), dtype=bool)
    cands_arr = np.zeros((ACTION_MAX_CANDIDATES, PIVOT_FEAT_DIM), dtype=np.float32)

    if cands_option == 'CP':
        candidates = get_pivots_cp(bin_obj)
    elif cands_option == 'EP':
        candidates = get_pivots_ep(bin_obj)
    elif cands_option == 'EMS':
        candidates = get_pivots_ems(bin_obj)
    else:  # 'EDP'
        candidates = Edge_Projection(bin_obj)

    limit = min(len(candidates), ACTION_MAX_CANDIDATES)
    bin_obj.pivotTree = PivotAVLTree()

    for i, pv in enumerate(candidates[:limit]):
        mask[i] = True
        bin_obj.pivotTree.insert(pv)
        cands_arr[i] = encode_pivot(pv, bin_obj)

    return mask, cands_arr

# ─────────────────────────────────────────────────────────────
# [수정] 통합된 rebuild_candidates
# ─────────────────────────────────────────────────────────────
def rebuild_candidates(TOTAL, preview_cnt, queue, bin, K, NOOP_IDX, cands_option='EDP', check=False) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    - preview_cnt: 보여줄 아이템 개수 (이 개수만큼 마스킹 계산)
    - 기존 selectable_len, preview_k 파라미터 삭제됨
    """
    mask  = np.zeros((TOTAL,), dtype=bool)
    cands = np.zeros((TOTAL, PIVOT_FEAT_DIM), dtype=np.float32)

    # 1. 큐에서 ID들 가져오기 (preview_cnt 만큼)
    head_ids = queue_head(preview_cnt, queue)
    
    # 2. ID를 이용해 최신 객체 가져오기
    look_pairs = get_look_items(head_ids)

    # 3. 마스크 생성
    # look_pairs에 있는 아이템들에 대해서만 루프가 돌기 때문에 
    # preview_cnt보다 큐 길이가 짧은 경우도 자동으로 처리됨.
    for slot, item in look_pairs:
        if check:
            m_i, c_i = get_action_mask(bin, item, cands_option) 
        else:
            m_i, c_i = get_action_mask_wo(bin, cands_option)

        start = slot * K
        end   = start + K
        mask[start:end]  = m_i
        cands[start:end] = c_i

    # (이전의 Preview Cutoff 루프는 더 이상 필요 없음)

    last_mask  = mask
    last_cands = cands
    
    # 모두 불가능하면 NOOP 허용
    if not bool(last_mask.any()):
        last_mask[NOOP_IDX] = True
    
    return last_mask, last_cands, [idx for (idx, _item) in look_pairs]

# ─────────────────────────────────────────────────────────────
# [수정] 통합된 place_by_action
# ─────────────────────────────────────────────────────────────
def place_by_action(action, preview_cnt, queue, bin_obj, last_mask, last_cands, K):
    slot, k = decode_action(action, K)

    # 1. 현재 슬롯에 해당하는 아이템 찾기
    head_ids = queue_head(preview_cnt, queue)
    
    # 슬롯 인덱스 유효성 체크
    if slot < 0 or slot >= len(head_ids):
        return False, "Invalid slot index"
    
    target_id = head_ids[slot]
    target_item = global_item_manager.get(target_id)
    
    if target_item is None:
        return False, "Item not found in manager"
    
    # 2. Action 유효성 확인
    idx_flat = slot * K + k
    if (last_mask is None or not last_mask[idx_flat]):
        return False, "Selected action is not feasible"

    # 3. Pivot 디코딩 & 배치 시도
    pivot_arr = last_cands[idx_flat]
    pivot = decode_pivot(pivot_arr, bin_obj)

    code, placed_item = checkPivot_R(
        bin_obj,
        target_item,
        [pivot.x, pivot.y, pivot.z],
        pivot.rt,
        apply_margin=True
    )
    if code != SUCCESS:
        return False, code
    else:
        placed_item.b_position = [pivot.x, pivot.y, pivot.z]
        placed_item.rotation_quat = pivot.rt
        
    # 4. 배치 확정
    global_item_manager.update(placed_item._id, placed_item)
    bin_obj.store(placed_item)
    try:
        queue.remove(target_id)
    except ValueError:
        pass # 이미 없는 경우 무시

    return True, placed_item