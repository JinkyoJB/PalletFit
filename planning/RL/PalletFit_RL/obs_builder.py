# planning/RL/PalletFit_RL/obs_builder.py
from __future__ import annotations
from typing import Dict, List, Optional, Deque
import numpy as np
from itertools import islice

from utils.util_functions import eps, clip01
from utils.overlap import overlap_intervals
from utils.get_value import (
    get_direction_overlap,
    score_ez_distribution,
    get_score_Guillotine,
    balance_feature,
    normalize_balance_features
)
from planning.item import RotationType
from planning.itemManager import global_item_manager

from planning.RL.PalletFit_RL.config import (
    OBS_TOPK_DEFAULT,
    ITEM_FEAT_DIM,
    GLOBAL_FEAT_DIM,
    PREVIEW_MAX,
    PIVOT_FEAT_DIM
)

# # ─────────────────────────────────────────────────────────────────────
# 공용 helper 함수
# ─────────────────────────────────────────────────────────────────────
def queue_head( n: int, queue) -> List[int]:
    """
    현재 큐의 헤더 n개를 반환.
    ★ 변경: 반환값은 item index가 아니라 item._id 리스트임.
    """
    n = max(0, min(n, len(queue)))
    return list(islice(queue, 0, n))

def get_look_items(head_ids: Optional[List[int]] = None):
    """
    큐의 앞쪽 ID들을 받아 실제 Item 객체로 변환.
    반환 형식: [(slot_idx, Item객체), ...]
    """
    if head_ids is None:
        return []
    
    look_pairs = []

    for slot_idx, iid in enumerate(head_ids):
        # ★ 핵심: 큐에 있는 ID로 Global Manager에서 '최신 상태' 객체를 가져옴
        real_item = global_item_manager.get(iid)
        
        # 방어 코드: 매니저에 없으면 items_list에서 찾기 (거의 그럴 일 없음)
        if real_item is None:
            # items_list를 순회하며 찾거나(느림), 에러 처리. 
            # 여기선 안전하게 None 무시 혹은 로그
            continue
            
        look_pairs.append((slot_idx, real_item))

    return look_pairs

# ─────────────────────────────────────────────────────────────────────
# Top-K 선택 (TopView 우선 → 부족분은 패딩)
# ─────────────────────────────────────────────────────────────────────
def select_topk_ids(bin_obj, k: int) -> List[Optional[int]]:
    """
    상단에서 보이는 아이템들을 ez(desc), x/y(asc), id(asc)로 정렬해 상위 k개 아이디(_id)만 반환.
    부족하면 None 패딩.
    """
    visible_items = bin_obj.get_visible_items_topdown() or []

    def _key_item(it):
        x, y, z = it.b_position
        return (-float(it.ez), float(x), float(y), int(it._id))  # ez desc, x/y asc, id asc

    visible_items.sort(key=_key_item)
    selected_ids = [it._id for it in visible_items[:k]]

    if len(selected_ids) < k:
        selected_ids += [None] * (k - len(selected_ids))
    else:
        selected_ids = selected_ids[:k]
    return selected_ids

# ─────────────────────────────────────────────────────────────────────
# 지지율 추정 (안전 처리: 값 범위/폴백)
# ─────────────────────────────────────────────────────────────────────
def estimate_support_ratio_safe(bin_obj, item) -> float:
    try:
        val = float(item.getBottomOverlap())
        if 0.0 <= val <= 1.0:
            return clip01(val)
        w, h, d = item.getDimension()
        base_area = max(eps(), float(w) * float(d))
        return clip01(val / base_area)
    except Exception:
        try:
            w, h, d = item.getDimension()
            base_area = max(eps(), float(w) * float(d))
            bottoms = bin_obj.get_bottom_items(item) or []
            x0, y0, _ = item.b_position
            x1, y1 = x0 + w, y0 + d
            area_sum = 0.0
            for b in bottoms:
                bx0, by0, _ = b.b_position
                bw, bh, bd = b.getDimension()
                bx1, by1 = bx0 + bw, by0 + bd
                ox = overlap_intervals(x0, x1, bx0, bx1)
                oy = overlap_intervals(y0, y1, by0, by1)
                if ox > 0 and oy > 0:
                    area_sum += ox * oy
            return clip01(area_sum / base_area)
        except Exception:
            return 0.0

# ─────────────────────────────────────────────────────────────────────
# 아이템 → [0,1] 정규화 피처 인코딩
# ─────────────────────────────────────────────────────────────────────
def _encode_item_from_obj(it, bin_obj) -> np.ndarray:
    """
    기존 encode_item(iid, bin_obj)와 동일한 스케일링을 '아이템 객체'로 수행.
    """
    W, H, D = float(bin_obj.width), float(bin_obj.height), float(bin_obj.depth)
    maxW = float(getattr(bin_obj, "max_weight", 0.0) or 1.0)

    if it is None:
        return np.zeros(ITEM_FEAT_DIM, dtype=np.float32)

    x, y, z = map(float, it.b_position)
    w, h, d = map(float, it.getDimension())
    ex, ey, ez = float(it.ex), float(it.ey), float(it.ez)

    # (1) 위치/크기 [0,1]
    cx = clip01((x + 0.5 * w) / eps(W))
    cy = clip01((y + 0.5 * d) / eps(D))
    cz = clip01((z + 0.5 * h) / eps(H))
    sx = clip01(w / eps(W))
    sy = clip01(d / eps(D))
    sz = clip01(h / eps(H))

    # (2) 무게/회전(2클래스)
    wfrac = clip01(float(getattr(it, "weight", 0.0)) / eps(maxW))
    rt_idx = RotationType.quat_to_index(getattr(it, "rotation_quat", None))

    # (3) 방향별 실제 갭/겹침
    try:
        dir_stats = get_direction_overlap(it, bin_obj, palletizing_mode=True)
    except Exception:
        dir_stats = {}

    def _safe(dir_name):
        return dir_stats.get(dir_name, (0.0, 0.0, 0.0, 0.0))

    gap_L, ra_L, rb_L, _ = _safe('left')
    gap_R, ra_R, rb_R, _ = _safe('right')
    gap_F, ra_F, rb_F, _ = _safe('front')
    gap_B, ra_B, rb_B, _ = _safe('back')

    gap_Ln = clip01(gap_L / eps(H))  # L/R → W
    gap_Rn = clip01(gap_R / eps(H))
    gap_Fn = clip01(gap_F / eps(W))  # F/B → H
    gap_Bn = clip01(gap_B / eps(W))

    # (4) z 보완 & 지지율
    top_z         = clip01(ez / eps(D))
    support_ratio = estimate_support_ratio_safe(bin_obj, it)

    feat_list = [
        cx, cy, cz, sx, sy, sz, wfrac,
        rt_idx,
        gap_Ln, ra_L, rb_L,
        gap_Rn, ra_R, rb_R,
        gap_Fn, ra_F, rb_F,
        gap_Bn, ra_B, rb_B,
        top_z, support_ratio,
    ]

    v = np.array(feat_list, dtype=np.float32)
    if v.shape[0] < ITEM_FEAT_DIM:
        v = np.pad(v, (0, ITEM_FEAT_DIM - v.shape[0]))
    elif v.shape[0] > ITEM_FEAT_DIM:
        v = v[:ITEM_FEAT_DIM]
    return v

# ─────────────────────────────────────────────────────────────────────
# 전역 스칼라
# ─────────────────────────────────────────────────────────────────────
def encode_globals(bin_obj) -> np.ndarray:
    def _squash_pos(x: float, alpha: float = 0.1) -> float:
        x = max(float(x), 0.0)
        return clip01(1.0 - np.exp(-alpha * x))

    su = clip01(float(getattr(bin_obj, "SU", 0.0)))

    try:
        h_score, c_score = score_ez_distribution(bin_obj, new_item=None)
        h_score = clip01(float(h_score))
        c_score = clip01(float(c_score))
    except Exception:
        h_score, c_score = 0.0, 0.0

    try:
        g_raw = float(get_score_Guillotine(bin_obj, loaded_item=None))
        g_s   = _squash_pos(g_raw, alpha=0.1)
    except Exception:
        g_s = 0.0

    try:
        imb_raw, I_raw, sum_m_t = balance_feature(bin_obj, cand_item=None)
        imb_n, I_n = normalize_balance_features(imb_raw, I_raw, sum_m_t)
    except Exception:
        imb_n, I_n = 0.0, 0.0

    return np.array([su, h_score, c_score, g_s, imb_n, I_n], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────
# 프리뷰/선택 영역 인코딩 (ID 기반 조회)
# ─────────────────────────────────────────────────────────────────────
def encode_preview_items_from_queue(bin_obj, queue: Deque[int], preview_cnt: int) -> np.ndarray:
    """
    PREVIEW_MAX(5) 행으로 고정된 프리뷰 텐서를 반환합니다.
    단, 실제 데이터는 min(preview_cnt, 큐 길이) 만큼만 채우고 나머지는 패딩합니다.
    """
    W, H, D = float(bin_obj.width), float(bin_obj.height), float(bin_obj.depth)
    maxW = float(getattr(bin_obj, "max_weight", 0.0) or 1.0)

    # [수정] preview_cnt만큼만 잘라내기 (단, PREVIEW_MAX를 넘을 순 없음)
    limit = min(preview_cnt, PREVIEW_MAX)
    head_ids = list(islice(queue, 0, limit))

    feats = []
    # 항상 PREVIEW_MAX(5) 길이만큼 반복 (부족하면 패딩)
    for idx in range(PREVIEW_MAX):
        if idx < len(head_ids):
            iid = head_ids[idx]
            it = global_item_manager.get(iid)
            
            if it is not None:
                w, h, d_dim = map(float, it.getDimension())
                wf = clip01(w / eps(W))
                hf = clip01(h / eps(H))
                df = clip01(d_dim / eps(D))
                wfraction = clip01(float(getattr(it, "weight", 0.0)) / eps(maxW))
                feats.append(np.array([wf, hf, df, wfraction], dtype=np.float32))
            else:
                # ID는 있는데 객체가 없는 경우 (0 패딩)
                feats.append(np.zeros((4,), dtype=np.float32))
        else:
            # 큐 데이터 부족 or preview_cnt 제한에 걸린 경우 (0 패딩)
            feats.append(np.zeros((4,), dtype=np.float32))
            
    return np.stack(feats, axis=0).astype(np.float32)


def build_observation_deque(bin_obj, queue: Deque[int], preview_cnt: int) -> Dict[str, np.ndarray]:
    """
    기존 관측 포맷 유지:
      - items_topk / items_mask / globals 그대로
      - preview_queue: queue(ID 리스트)를 사용하여 생성
    """
    # 초기 상태 대응
    if bin_obj is None:
        return {
            "items_topk":    np.zeros((OBS_TOPK_DEFAULT, ITEM_FEAT_DIM), dtype=np.float32),
            "items_mask":    np.zeros((OBS_TOPK_DEFAULT,), dtype=np.float32),
            "globals":       np.zeros((GLOBAL_FEAT_DIM,), dtype=np.float32),
            "preview_queue": np.zeros((PREVIEW_MAX, 4), dtype=np.float32),
        }

    ids: List[Optional[int]] = select_topk_ids(bin_obj, OBS_TOPK_DEFAULT)

    feats: List[np.ndarray] = []
    mask_vals: List[float] = []
    for iid in ids:
        it = None if iid is None else global_item_manager.get(iid)
        feats.append(_encode_item_from_obj(it, bin_obj))
        mask_vals.append(0.0 if iid is None else 1.0)

    if len(feats) == 0:
        items_topk = np.zeros((OBS_TOPK_DEFAULT, ITEM_FEAT_DIM), dtype=np.float32)
        items_mask = np.zeros((OBS_TOPK_DEFAULT,), dtype=np.float32)
    else:
        items_topk = np.stack(feats, axis=0).astype(np.float32)
        items_mask = np.array(mask_vals, dtype=np.float32)

    globals_vec = encode_globals(bin_obj)
    
    # ★ 수정된 함수 호출 (item_list는 이제 선택사항)
    preview_items_vec = encode_preview_items_from_queue(bin_obj, queue, preview_cnt)
    
    return {
        "items_topk":     items_topk,
        "items_mask":     items_mask,
        "globals":        globals_vec,
        "preview_queue":  preview_items_vec,
    }

def make_obs_space_gym():
    try:
        from gymnasium import spaces
    except Exception:
        return None
    return spaces.Dict({
        "items_topk":     spaces.Box(low=0.0, high=1.0, shape=(OBS_TOPK_DEFAULT, ITEM_FEAT_DIM), dtype=np.float32),
        "items_mask":     spaces.Box(low=0.0, high=1.0, shape=(OBS_TOPK_DEFAULT,), dtype=np.float32),
        "globals":        spaces.Box(low=0.0, high=1.0, shape=(GLOBAL_FEAT_DIM,), dtype=np.float32),
        "preview_queue":  spaces.Box(0.0, 1.0, shape=(PREVIEW_MAX, 4), dtype=np.float32),
    })

def build_obs(head_q, bin_obj, last_mask, last_cands, TOTAL, preview_cnt: int) -> Dict[str, np.ndarray]:
    """
    [수정] preview_cnt 인자 추가
    """
    obs = build_observation_deque(
        bin_obj=bin_obj,
        queue=head_q,
        preview_cnt=preview_cnt, # 전달
    )
    act_mask = last_mask if last_mask is not None \
            else np.zeros((TOTAL), dtype=bool)
    act_cands = last_cands if last_cands is not None \
                else np.zeros((TOTAL, PIVOT_FEAT_DIM), dtype=np.float32)

    obs["act_mask"]  = act_mask.astype(np.float32)
    obs["act_cands"] = act_cands.astype(np.float32)
    return obs