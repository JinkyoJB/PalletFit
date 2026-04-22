# utils/get_value.py
from planning.item import RotationType
from planning.itemManager import global_item_manager
from .overlap import overlap_intervals, overlap_length, compute_overlap_area
from .constants import BIG_POS, BIG_NEG, EPS
import copy, math
import numpy as np
from collections import Counter,defaultdict
from typing import Dict, List, Tuple

def get_dim_for_rt(it, rt):
    '''
    주어진 회전에서 (w,h,d)를 구하는 보조 함수
    '''
    tmp = copy.deepcopy(it)
    tmp.rotation_quat = rt
    return tmp.getDimension()


def get_bounds(vertices):
    """
    주어진 직육면체의 정점(vertices)으로부터 축별 최소값과 최대값을 계산합니다.
    
    Parameters:
    - vertices (list of lists): 직육면체의 8개 정점 [[x, y, z], ...]
    
    Returns:
    - tuple: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def get_distance(p1, p2):
    p1 = [float(p) for p in p1]
    p2 = [float(p) for p in p2]
    d2 = sum((p1[i] - p2[i]) ** 2 for i in range(3))   # 제곱합
    return d2 ** 0.5


def get_x_absolute(p1, p2):
    """
    두 점 p1, p2 사이의 x축 절대값 거리를 계산합니다.
    """
    p1 = p1[0]
    p2 = p2[0]

    return abs(p1 - p2)

def get_center_item(item):
    """
    Item 객체의 중심점을 계산합니다.
    """
    w,h,d = item.getDimension()
    x,y,z = item.b_position
    return [x + w/2, y + h/2, z + d/2]


def get_distance_item(item1, item2):
    """
    두 Item 객체 사이의 거리를 계산합니다.
    
    Parameters:
    - item1, item2 (Item): 두 Item 객체
    """
    return get_distance(item1.b_position, item2.b_position)



def choose_best_rotation(item, ref_item, placement_mode='vertical'):

    """
    item: 현재 배치하려는 아이템
    ref_item: 기준이 되는 아이템 (폭/높이 비교 대상)
    placement_mode:
    - 'horizontal': 왼·오 배치 → 'height' 비교
    - 'vertical':   앞·뒤 배치 → 'width' 비교
    - 그 외 확장 가능 (예: 'stack'으로 depth 비교 등)
    
    회전 후보: (현재 item.rotation_quat) & 그 짝꿍
    
    비교 기준:
    - horizontal -> item.height vs ref_item.height
    - vertical   -> item.width vs ref_item.width
    """
    # ================== 보조 함수 ==================

    # 함수: 주어진 회전에서 '비교할 아이템 치수' 추출
    def get_compare_val(it, rt, placement_mode='vertical'):
        w, h, d = get_dim_for_rt(it, rt)
        if placement_mode == 'horizontal':
            return h
        else:  # 'vertical'
            return w


    # 참조 아이템의 dimension (기본 회전)
    ref_w, ref_h, ref_d = ref_item.getDimension()

    # 실제로 비교할 '기준값' 뽑기
    if placement_mode == 'horizontal':
        # 왼·오 배치 -> 높이(height) 비교
        ref_val = ref_h
    else:
        # 기본값: 'vertical' -> 폭(width) 비교
        ref_val = ref_w

    # 회전 후보
    r0 = item.rotation_quat
    r1 = RotationType.get_rotation_pair(r0)

    # r0에 대한 차이
    val0 = get_compare_val(item, r0, placement_mode)
    diff0 = abs(val0 - ref_val)

    # 짝꿍이 없으면 r0 반환
    if r1 is None:
        return r0

    # r1에 대한 차이
    val1 = get_compare_val(item, r1, placement_mode)
    diff1 = abs(val1 - ref_val)

    # 더 작은 쪽 우선
    return r0 if diff0 <= diff1 else r1

def get_direction_overlap(item, bin, palletizing_mode: bool = False):
    """
    bin.binIndex를 활용해 item의 방향별 gap 및 겹침 비율을 계산.
    """
    if 'direction_overlap' not in item.options:
        item.options['direction_overlap'] = {}

    for dir_ in ['left', 'right', 'front', 'back']:
        # if item._id == 40 and bin.size ==3:
        #     print(f"get_direction_overlap: {dir_}")
        gap, ratioA, ratioB, sum_overlap = direction_gap_rtree(bin, item, dir_, palletizing_mode)
        # 2자리 유효숫자로 round
        gap = round(gap, 3)
        ratioA = round(ratioA, 3)
        ratioB = round(ratioB, 3)
        sum_overlap = round(sum_overlap, 3)
        item.options['direction_overlap'][dir_] = (gap, ratioA, ratioB, sum_overlap)

    return item.options['direction_overlap']

def direction_gap_rtree(
        bin, item, direction, palletizing_mode):
    """
    (gap , ov_ratioA , ov_ratioB , ov_len)

      · gap        : item 과 방향쪽 가장 가까운 물체(또는 벽)와의 거리
      · overlap     : (y 또는 x)축 겹친 면적들의 '총합'
      · ov_ratioA  :  item(A) 단면(Aw*Ad 또는 Ah*Ad) 기준 총 겹친 면적 비율
      · ov_ratioB  :  이웃 B 들 단면 총합 기준 총 겹친 면적 비율

      · palletizing_mode=True  ➜ 벽-접촉은 누적하지 않음
      · palletizing_mode=False ➜ 기존 bin-packing 로직 유지    
    """
    # ───────── 벽 붙음 처리 공통 헬퍼 ──────────
    def _touch_wall(face: str, z_min=0, z_max=BIG_POS):
        """
        left  → 벽 면적 = Bin.height × Bin.depth
        front → 벽 면적 = Bin.width  × Bin.depth
        right → 벽 면적 = Bin.height × Bin.depth
        back  → 벽 면적 = Bin.width  × Bin.depth
        """
        nonlocal sum_len, sum_overlap, sum_B_area, min_gap
        if palletizing_mode:
            return 0.0         # 벽 면적도 0 으로 간주
        if face == "left":  # 왼쪽 벽
            wall_area = bin.height * bin.depth
            cand_ids = bin.search_xyz(BIG_NEG, EPS, BIG_NEG, bin.height, z_min, z_max)
            # 후보아이템들로 변환
            cand_items = [global_item_manager.get(iid) for iid in cand_ids]
            # b_position[1]이 작은 순으로 정렬
            cand_items.sort(key=lambda B: B.b_position[1])

            for i, it in enumerate(cand_items):
                if it is None or it is item:
                    continue
                area = (it.getDimension()[1] + bin.margin_y*(i+1 if not len(cand_items)==(i+1) else 0)) * it.getDimension()[2]
                wall_area -= area

            sum_len     = Ah            # y-축 전체가 맞닿음
            sum_overlap = Ah * Ad
        elif face == "right":  # 오른쪽 벽
            wall_area = bin.height * bin.depth
            cand_ids = bin.search_xyz(bin.width-EPS, BIG_POS, BIG_NEG, bin.height, z_min, z_max)
            # 후보아이템들로 변환
            cand_items = [global_item_manager.get(iid) for iid in cand_ids]
            # b_position[1]이 작은 순으로 정렬
            cand_items.sort(key=lambda B: B.b_position[1])
            for i, it in enumerate(cand_items):
                if it is None or it is item:
                    continue
                area = (it.getDimension()[1] + bin.margin_y*(i+1 if not len(cand_items)==(i+1) else 0)) * it.getDimension()[2]
                wall_area -= area
            sum_len     = Ah            # y-축 전체가 맞닿음
            sum_overlap = Ah * Ad
        elif face == "front":  # 앞 벽
            wall_area = bin.width * bin.depth
            cand_ids = bin.search_xyz(BIG_NEG, bin.width, BIG_NEG, EPS, z_min, z_max)
            # 후보아이템들로 변환
            cand_items = [global_item_manager.get(iid) for iid in cand_ids]
            # b_position[0]이 작은 순으로 정렬
            cand_items.sort(key=lambda B: B.b_position[0])
            for i, it in enumerate(cand_items):
                if it is None or it is item:
                    continue
                area = (it.getDimension()[0] + bin.margin_x*(i+1 if not len(cand_items)==(i+1) else 0)) * it.getDimension()[2]
                wall_area -= area
            sum_len     = Aw            # x-축 전체가 맞닿음
            sum_overlap = Aw * Ad
        elif face == "back":   # 뒤 벽
            wall_area = bin.width * bin.depth
            cand_ids = bin.search_xyz(BIG_NEG, bin.width, bin.height-EPS, BIG_POS, z_min, z_max)
            # 후보아이템들로 변환
            cand_items = [global_item_manager.get(iid) for iid in cand_ids]
            # b_position[0]이 작은 순으로 정렬
            cand_items.sort(key=lambda B: B.b_position[0])
            for i, it in enumerate(cand_items):
                if it is None or it is item:
                    continue
                area = (it.getDimension()[0] + bin.margin_x*(i+1 if not len(cand_items)==(i+1) else 0)) * it.getDimension()[2]
                wall_area -= area
            sum_len     = Aw            # x-축 전체가 맞닿음
            sum_overlap = Aw * Ad

        sum_B_area  = wall_area         
        return wall_area                # 벽 면적 반환

    mx, my = bin.margin_x+EPS, bin.margin_y+EPS

    Ax, Ay, Az = item.b_position
    Aw, Ah, Ad = item.getDimension()
    Ax2, Ay2, Az2 = Ax+Aw, Ay+Ah, Az+Ad
    z_min, z_max = Az, Az2

    min_gap      = None      # 가장 가까운 gap
    sum_overlap  = 0.0       # A와 B 들이 실제로 겹친 면적 총합
    sum_len      = 0.0       # (y/x) 방향 겹친 길이 총합
    sum_B_area   = 0.0       # B 들 단면(방향별) 면적 총합

    # ───────────────────────── 내부 helper ─────────────────────────
    def _accumulate(gap_val, over_len_axis, over_z, B_dim_face, *, thr):
        """
        gap 이 thr 이내일 때만 누적.
        """
        nonlocal sum_overlap, sum_len, sum_B_area
        if gap_val <= thr:
            sum_len     += over_len_axis
            sum_overlap += over_len_axis * over_z
            sum_B_area  += B_dim_face

    # ---------------------------------------------------------
    if direction == "left":
        cand_ids = bin.search_xyz(BIG_NEG, Ax, Ay, Ay2, Az, Az2)
        # 후보아이템들로 변환
        cand_items = [global_item_manager.get(iid) for iid in cand_ids]
        # b_position[1]이 작은 순으로 정렬
        cand_items.sort(key=lambda B: B.b_position[1])
        for i, B in enumerate(cand_items):
            if B is None or B._id is item._id:
                continue
            _ , B_height, B_depth = B.getDimension()

            if not overlap_intervals(Ay, Ay2, B.b_position[1], B.ey):  continue
            if not overlap_intervals(Az, Az2, B.b_position[2], B.ez):     continue
            if B.ex > Ax:   # 진짜 왼쪽이 아님
                continue

            gap = Ax - B.ex
            min_gap = gap if min_gap is None else min(min_gap, gap)

            yov = overlap_length(Ay, Ay2, B.b_position[1], B.ey + bin.margin_y*(i+1 if not len(cand_items)==(i+1) else 0))
            zov = overlap_length(Az, Az2, B.b_position[2], B.ez)

            _accumulate(gap, yov, zov, B_height * B_depth, thr=mx)          # ← 조건 적용
        if min_gap is None:                 # 벽(Left)까지 거리만 남음
            min_gap = Ax
            if min_gap <= mx*2:             # “거의 붙어있다” 판정
                _touch_wall(direction, z_min, z_max)           # 벽 면적 사용
        ov_ratioA = min(sum_overlap / (Ah * Ad+1e-8), 1.0)
        if sum_B_area < EPS:                 # EPS = 1e-8 정도의 작은 값
            ov_ratioB = 0.0
        else:
            ov_ratioB = min(sum_overlap / sum_B_area + EPS, 1.0)
    # ---------------------------------------------------------
    elif direction == "right":
        cand_ids = bin.search_xyz(Ax2, BIG_POS, Ay, Ay2, Az, Az2)
        # 후보아이템들로 변환
        cand_items = [global_item_manager.get(iid) for iid in cand_ids]
        # b_position[1]이 작은 순으로 정렬
        cand_items.sort(key=lambda B: B.b_position[1])
        for i, B in enumerate(cand_items):
            if B is None or B._id is item._id:
                continue
            _ , B_height, B_depth = B.getDimension()

            if not overlap_intervals(Ay, Ay2, B.b_position[1], B.ey):  continue
            if B.b_position[0] < Ax2:   continue

            gap = B.b_position[0] - Ax2
            min_gap = gap if min_gap is None else min(min_gap, gap)

            yov = overlap_length(Ay, Ay2, B.b_position[1], B.ey + bin.margin_y*(i+1 if not len(cand_items)==(i+1) else 0))
            zov = overlap_length(Az, Az2, B.b_position[2], B.ez)
            _accumulate(gap, yov, zov,
                        B_height * B_depth,
                        thr=mx)
        if min_gap is None:
            min_gap      = bin.width - Ax2
            if min_gap <= mx*2:
                _touch_wall(direction, z_min, z_max)           # 벽 면적 사용
        ov_ratioA = min(sum_overlap / (Ah * Ad+1e-8), 1.0)
        if sum_B_area < EPS:                 # EPS = 1e-8 정도의 작은 값
            ov_ratioB = 0.0
        else:
            ov_ratioB = min(sum_overlap / sum_B_area + EPS, 1.0)
    # ---------------------------------------------------------
    elif direction == "front":
        cand_ids = bin.search_xyz(Ax, Ax2, BIG_NEG, Ay, z_min, z_max)
        # 후보아이템들로 변환
        cand_items = [global_item_manager.get(iid) for iid in cand_ids]
        # b_position[0]이 작은 순으로 정렬
        cand_items.sort(key=lambda B: B.b_position[0])
        for i, B in enumerate(cand_items):
            if B is None or B._id is item._id:
                continue
            B_width, _ , B_depth = B.getDimension()

            if not overlap_intervals(Ax, Ax2, B.b_position[0], B.ex):  continue
            if B.ey > Ay:   continue

            gap = Ay - B.ey
            min_gap = gap if min_gap is None else min(min_gap, gap)

            xov = overlap_length(Ax, Ax2, B.b_position[0], B.ex + bin.margin_x*(i+1 if not len(cand_items)==(i+1) else 0))
            zov = overlap_length(Az, Az2, B.b_position[2], B.ez)
            _accumulate(gap, xov, zov,
                        B_width * B_depth,
                        thr=my)
        if min_gap is None:
            min_gap      = Ay
            if min_gap <= my*2:
                _touch_wall(direction, z_min, z_max)           # 벽 면적 사용
        ov_ratioA = min(sum_overlap / (Aw * Ad+1e-8), 1.0)
        if sum_B_area < EPS:                 # EPS = 1e-8 정도의 작은 값
            ov_ratioB = 0.0
        else:
            ov_ratioB = min(sum_overlap / sum_B_area + EPS, 1.0)
    # ---------------------------------------------------------
    elif direction == "back":
        cand_ids = bin.search_xyz(Ax, Ax2, Ay2, BIG_POS, z_min, z_max)
        # 후보아이템들로 변환
        cand_items = [global_item_manager.get(iid) for iid in cand_ids]
        # b_position[0]이 작은 순으로 정렬
        cand_items.sort(key=lambda B: B.b_position[0])
        for i, B in enumerate(cand_items):
            if B is None or B._id is item._id:
                continue
            B_width, _ , B_depth = B.getDimension()
            if not overlap_intervals(Ax, Ax2, B.b_position[0], B.ex):  continue
            if B.b_position[1] < Ay2:   continue

            gap = B.b_position[1] - Ay2
            min_gap = gap if min_gap is None else min(min_gap, gap)

            xov = overlap_length(Ax, Ax2, B.b_position[0], B.ex + bin.margin_x*(i+1 if not len(cand_items)==(i+1) else 0))
            zov = overlap_length(Az, Az2, B.b_position[2], B.ez)
            _accumulate(gap, xov, zov,
                        B_width * B_depth,
                        thr=my)
        if min_gap is None:
            min_gap      = bin.height - Ay2
            if min_gap <= my*2:
                _touch_wall(direction, z_min, z_max)           # 벽 면적 사용
        ov_ratioA = min(sum_overlap / (Aw * Ad+1e-8), 1.0)
        if sum_B_area < EPS:                 # EPS = 1e-8 정도의 작은 값
            ov_ratioB = 0.0
        else:
            ov_ratioB = min(sum_overlap / sum_B_area + EPS, 1.0)
    else:
        raise ValueError("direction must be 'left'/'right'/'front'/'back'")

    # gap 은 항상 ≥0
    return float(min_gap), float(ov_ratioA), float(ov_ratioB), float(sum_overlap)

# -------------------------------------------------------------------
# bench_bin에 new_item을 배치했을 때 얻을 수 있는 수치를 반환해주는 헬퍼 함수
# -------------------------------------------------------------------
def flat_surface_weight(bin, new_item) -> int:
    """
    상단(ez 최대)에 놓여 있는 아이템들의 윗면 z(ez) 값과
    `new_item.ez` 가 얼마나 많이 일치하는지를 센다.

    반환값
    -------
    match_cnt : int
        ▸ 0  →  같은 높이(ez)를 가진 상단 아이템이 없음  
        ▸ k  →  new_item 을 올려두면 k 개의 상단 아이템과
                 정확히 같은 높이(± tol)로 평평한 면을 이룰 수 있음
    """
    # ── 1) 현재 ‘맨 위에 있는’ 아이템 id 모으기 ─────────────────────
    try:
        top_items: list[int] = bin.get_visible_items_topdown()   # 메서드가 있으면 사용
    except AttributeError:
        top_items = [it._id for it in getattr(bin, "get_visible_items_topdown")()]

    # ── 2) 상단 아이템들의 ez 값을 리스트로 수집 (중복 허용) ──────────
    flat_ezs: list[float] = [i.ez for i in top_items if i is not None and i.ez is not None]

    # ── 3) new_item 과 tol 이내로 같은 ez 가 몇 개인지 카운트 ───────
    new_ez = round(new_item.ez, 3)
    match_cnt = sum(1 for ez0 in flat_ezs if abs(new_ez - ez0) <= EPS+5)

    return match_cnt


# ──────────────────────────────────────────────
# 1) ez-별 개수 & 면적을 한 번에 수집
# ──────────────────────────────────────────────
def collect_ez_stats(
    bin,
    new_item=None,
    *,
    round_ndigits: int = 2
) -> Tuple[Dict[float, int], Dict[float, float]]:
    """
    Returns
    -------
    counts : Dict[ez, count]
    areas  : Dict[ez, total_xy_area]
    """
    counts = Counter()
    areas  = defaultdict(float)

    top_items = bin.get_visible_items_topdown()
    if not top_items and new_item is None:
        return {}, {}

    # (1) 기존 아이템
    for it in top_items:
        ez = round(float(it.ez), round_ndigits)
        counts[ez] += 1
        areas[ez]  += float(it.width) * float(it.height)

    # (2) 후보 아이템
    if new_item is not None:
        ez = round(float(new_item.ez), round_ndigits)
        counts[ez] += 1
        areas[ez]  += float(new_item.width) * float(new_item.height)

    return dict(counts), dict(areas)

# ──────────────────────────────────────────────
# 2) Height / Cluster 점수 계산
# ──────────────────────────────────────────────
def score_ez_distribution(
    bin,
    new_item=None,
    *,
    round_ndigits: int = 3,
    Y_height: float = 1.0,
    Y_cluster: float = 1.0
) -> Tuple[float, float]:
    """
    Returns
    -------
    height_score, cluster_score
    """
    counts, areas = collect_ez_stats(bin, new_item, round_ndigits=round_ndigits)
    if not counts:
        return 0.0, 0.0

    # ── (A) Height Score ─────────────────────────────────────
    total_items = sum(counts.values())
    avg_ez      = sum(ez * c for ez, c in counts.items()) / total_items
    h_raw       = 1.0 - (avg_ez / bin.depth)            # 0~1
    height_score = max(0.0, min(h_raw, 1.0)) ** Y_height

    # ── (B) Cluster Score  (면적 비율) ──────────────────────
    max_area   = max(areas.values())
    bin_area   = float(bin.width) * float(bin.height)
    cluster_score = min(max_area / bin_area, 1.0) ** Y_cluster   # 0~1

    return height_score, cluster_score


def get_score_SameDepth_in_graph(bin, loaded_item) -> int:
    """
    (1) loaded_item 을 bin에 *임시* 등록
    (2) 좌·우·앞·뒤 이웃 중 ez 가 거의 같은 아이템 수를 카운트
    (3) 그래프·인덱스에서 깨끗하게 제거
    """
    iid = loaded_item._id
    # ─── A. 임시로 등록 ──────────────────────
    # loaded_item._id에 해당하는 global_item_manager 아이템을 가져옴
    bk_item = global_item_manager.get(iid)
    if bk_item is None:
        raise ValueError(f"Item with ID {iid} not found in global_item_manager.")
    
    # loaded_item 정보를 global_item_manager에 업데이트
    global_item_manager.update(iid, loaded_item)

    # bin에 loaded_item 저장
    bin.store(loaded_item)

    # ─── B. 점수 계산 ───────────────────────────────────────────────────
    gnode = bin.graph.get(iid, None)

    target_ez = float(loaded_item.ez)
    score = 0
    for d in ("left", "right", "front", "back"):
        for nbr_id in gnode[d]:
            nbr = global_item_manager.get(nbr_id)
            if nbr and abs(float(nbr.ez) - target_ez) <= EPS:
                score += 1

    # ─── C. 깨끗이 정리 ─────────────────────────────────────────────────
    bin.remove(iid)

    return score

def get_score_Guillotine(bin, loaded_item=None):
    """
    Guillotine score를 계산한다.
    - loaded_item 이 주어지면: bin + loaded_item(임시삽입) 상태에서,
      loaded_item의 z-슬랩 안의 아이템들만 대상으로 집계.
    - loaded_item 이 None이면: bin 내 모든 아이템을 대상으로,
      서로 z-구간이 겹치는 (동일 슬랩) 쌍에 한해 집계.

    반환: float (log(1 + exp(0.2 * (H+V))))
    """
    temp_insert = False
    bk_item = None

    try:
        tol_x = bin.margin_x + EPS
        tol_y = bin.margin_y + EPS

        # ----- 대상 z-범위 설정 & (필요시) loaded_item 임시 삽입 -----
        if loaded_item is not None:
            lid = loaded_item._id
            bk_item = global_item_manager.get(lid)
            # 아직 bin에 없으면 임시 삽입
            if lid not in getattr(bin, "xy_map", {}):
                bin._insert_rtree(lid, loaded_item)
                temp_insert = True
                global_item_manager.update(lid, loaded_item)

            z_min = float(loaded_item.b_position[2])
            z_max = float(loaded_item.ez)
        else:
            # bin 전체를 본다
            z_min = 0.0
            z_max = float(bin.depth)

        # ----- 1) z-범위 내 아이템 후보 수집 -----
        ids = bin.search_xyz(
            0.0, float(bin.width),
            0.0, float(bin.height),
            z_min, z_max
        )

        horiz, vert = [], []

        # ----- 2) 쌍별로 seam 후보 수집 (z-오버랩 필수) -----
        for i, ida in enumerate(ids):
            a = global_item_manager.get(ida)
            if a is None:
                continue
            ax0, ay0, _ = a.b_position
            ax1, ay1 = a.ex, a.ey
            az0, az1 = a.b_position[2], a.ez

            # x,y 근방 후보
            cand = bin.search_xy(
                ax0 - max(tol_x, tol_y),
                ax1 + max(tol_x, tol_y),
                ay0 - max(tol_x, tol_y),
                ay1 + max(tol_x, tol_y),
            )

            for idb in cand:
                if idb <= ida:  # (A,B) 중복 방지
                    continue
                b = global_item_manager.get(idb)
                if b is None:
                    continue
                bx0, by0, _ = b.b_position
                bx1, by1 = b.ex, b.ey
                bz0, bz1 = b.b_position[2], b.ez

                # z-슬랩 겹침이 없으면 스킵 (loaded_item=None일 때 필수)
                if min(az1, bz1) - max(az0, bz0) <= EPS:
                    continue

                # ─ Horizontal (y 고정, x 뻗음): y가 맞닿고 x가 겹치면 세그먼트 추가
                if abs(ay1 - by0) <= tol_y or abs(by1 - ay0) <= tol_y:
                    y_fix = ay1 if abs(ay1 - by0) <= abs(by1 - ay0) else by1
                    x0, x1 = max(ax0, bx0), min(ax1, bx1)
                    if x1 - x0 > EPS:
                        horiz.append((round(y_fix, 3), x0, x1))

                # ─ Vertical (x 고정, y 뻗음): x가 맞닿고 y가 겹치면 세그먼트 추가
                if abs(ax1 - bx0) <= tol_x or abs(bx1 - ax0) <= tol_x:
                    x_fix = ax1 if abs(ax1 - bx0) <= abs(bx1 - ax0) else bx1
                    y0, y1 = max(ay0, by0), min(ay1, by1)
                    if y1 - y0 > EPS:
                        vert.append((round(x_fix, 3), y0, y1))

        # ----- 3) 동일 y(x) 선상의 세그먼트 병합 → 개수 세기 -----
        def merge_and_count(seg_list, tol):
            if not seg_list:
                return 0
            buckets = defaultdict(list)
            for c, s, e in seg_list:
                buckets[c].append((s, e))
            cnt = 0
            for intervals in buckets.values():
                intervals.sort()
                cur_s, cur_e = intervals[0]
                for s, e in intervals[1:]:
                    if s <= cur_e + tol:
                        cur_e = max(cur_e, e)
                    else:
                        cnt += 1
                        cur_s, cur_e = s, e
                cnt += 1
            return cnt

        h_cnt = merge_and_count(horiz, tol_x)
        v_cnt = merge_and_count(vert, tol_y)
        total_cnt = h_cnt + v_cnt

        # 스무딩된 점수 (기존 형태 유지)
        score = math.log(1.0 + math.exp(0.2 * total_cnt))
        return score

    finally:
        # 임시 삽입 원복
        if temp_insert and loaded_item is not None:
            try:
                bin._delete_rtree(loaded_item._id)
            finally:
                global_item_manager.update(loaded_item._id, bk_item)


def distance_sq_from_origin(ib):
    x,y,z = ib.b_position
    dist_sq = x**2 + y**2 + z**2
    return dist_sq


def get_top_face_occupancy(bin, item):
    '''
    입력된 아이템의 bottom 노드들의 top_face_occupancy를 업데이트
    '''
    bottom_list = bin.get_bottom_items_in_graph(item)
    if bottom_list:
        for bottom_item in bottom_list:
            if bottom_item.top_face_occupancy <= 0:
                continue
            # top_face_occupancy 업데이트
            pivot_bounds = item.getFaceInfo('bottom')[1]
            pivot_bounds =(
                pivot_bounds['x'][0], pivot_bounds['x'][1],
                pivot_bounds['y'][0], pivot_bounds['y'][1],
                pivot_bounds['z'][0], pivot_bounds['z'][1]
            )

            item_bounds = bottom_item.getFaceInfo('top')[1]
            item_bounds = (
                item_bounds['x'][0], item_bounds['x'][1],
                item_bounds['y'][0], item_bounds['y'][1],
                item_bounds['z'][0], item_bounds['z'][1]
            )
            overlap_area = compute_overlap_area(pivot_bounds, item_bounds)
            bottom_item.top_face_occupancy -= overlap_area
    
    # item.children이 있다면 재귀 호출
    if item.children_ids:
        for child_id in item.children_ids:
            child_item = global_item_manager.get(child_id)
            get_top_face_occupancy(bin, child_item)


def weight_balance_score(
    bin,
    cand_item=None,
    *,
    dist_weight: float = 1.0,   # ① COM 거리 계수 (mm·kg)
    lr_weight : float = 1.0,   # ② 좌·우 불균형 계수 (kg)
    fb_weight : float = 1.0    # ③ 앞·뒤 불균형 계수 (kg)
) -> float:
    """
    ▸ (정규화 X)   값이 **클수록** 중앙에서 멀고, 치우쳐 있다.
    ▸ 스코어 구성
        S = w_d * (COM 거리, mm)         +
            w_lr* |왼쪽 W − 오른쪽 W|   +
            w_fb* |앞쪽 W − 뒤쪽 W|
    """
    # ───────── 1) 아이템 목록 & 무게·좌표 수집 ──────────
    items = [global_item_manager.get(i) for i in bin.item_ids]
    if cand_item is not None:
        items.append(cand_item)

    xs, ys, ws = [], [], []
    for it in items:
        if it is None:                # 드물게 None 이 들어올 때 방지
            continue
        ix, iy, _ = it.b_position
        iw, ih, _ = it.getDimension()
        xs.append(ix + iw * 0.5)      # 무게중심 (x,y)
        ys.append(iy + ih * 0.5)
        ws.append(it.weight)

    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
    total_w    = ws.sum() + 1e-8      # 0 div 방지

    # ───────── 2) 전체 Center of Mass → bin 중앙 거리(mm) ──────────
    cx_bin, cy_bin = bin.width * 0.5, bin.height * 0.5
    com_x = (xs * ws).sum() / total_w
    com_y = (ys * ws).sum() / total_w
    d_com = math.hypot(com_x - cx_bin, com_y - cy_bin)   # mm

    # ───────── 3) 축별 무게 차(kg) ──────────
    left_w  = ws[xs < cx_bin].sum()
    right_w = total_w - left_w
    front_w = ws[ys < cy_bin].sum()    # y 작음 = bin 앞
    back_w  = total_w - front_w
    lr_diff = abs(left_w  - right_w)   # kg
    fb_diff = abs(front_w - back_w)    # kg

    # ───────── 4) 비정규화 종합 스코어 ──────────
    return (dist_weight * d_com) + (lr_weight * lr_diff) + (fb_weight * fb_diff)


def weight_std_score_old(
    bin,
    cand_item=None,
    grid_n: int = 3,
    center_bias: float = 1.0,    # ← 중앙 가중치 (B)
    alpha: float = 0.85,          # 표준편차 비중
    beta: float = 0.15,       # 모멘트 비중  (A)
    gamma=1.4,                  # ★ 확장 지수
) -> float:
    '''
    center_bias: 중앙에 곱해지는 가중치. 해당 파라미터가 작을수록 작은 아이템이 놓인다고 생각하기 때문에 중앙 집중화됨.
    '''

    # 1) 구분선
    xs = [bin.width  * i / grid_n for i in range(1, grid_n)]
    ys = [bin.height * i / grid_n for i in range(1, grid_n)]

    # 2) 버킷
    w_grid = np.zeros((grid_n, grid_n), dtype=float)

    def _add(it, w):
        x, y, _ = it.b_position
        col = sum(x >= t for t in xs)
        row = sum(y >= t for t in ys)

        # ── B. 중앙 가중치
        if row == grid_n // 2 and col == grid_n // 2:
            w *= center_bias
        w_grid[row, col] += w

        # ── A. 모멘트 누적
        it_w, it_h, it_d = it.getDimension()
        dx = (x + it_w  * 0.5) - bin.width  * 0.5
        dy = (y + it_h * 0.5) - bin.height * 0.5
        moments.append(w * math.hypot(dx, dy))  # r = √(dx²+dy²)

    moments = []          # 모멘트 누적용

    for iid in bin.item_ids:
        it = global_item_manager.get(iid)
        if it is not None:
            _add(it, it.weight)

    if cand_item is not None:
        _add(cand_item, cand_item.weight)

    # 3) 표준편차
    flat = w_grid.flatten()
    std  = np.std(flat)
    var  = std ** 2            # ★ 분산(σ²) 사용

    # 4) 모멘트
    mom  = sum(moments) / (sum(flat) + 1e-8)   # kg·mm  → 무게당 평균 거리
    raw = alpha * var + beta * mom
    score = (raw + 1e-9) ** gamma        # γ>1  -> 작아지면 큰 보너스

    # 5) 최종 점수
    return score

def weight_std_score(
    bin,
    cand_item=None,
    *,
    cell_weight = None,   # (3,3) 또는 (9,)  실수 배열
    alpha: float = 0.85,
    beta:  float = 0.15,
    gamma: float = 1.4,
) -> float:
    """
    • 그리드 크기는 3 × 3 고정.
    • `cell_weight[row, col]` : (0,0)=좌하단  ↔  (2,2)=우상단.
       └ 값이 클수록 '해당 셀에 무게가 모이면 좋다'고 간주.
       └ 주어지지 않으면 모든 셀이 1.0 (균등).
    """
    grid_n = 3
    # 0) 가중치 행렬 준비 -------------------------------------------------
    if cell_weight is None:
        cell_weight = np.ones((grid_n, grid_n), dtype=float)
    else:
        cell_weight = np.asarray(cell_weight, dtype=float).reshape(grid_n, grid_n)

    # 1) 구분선 ----------------------------------------------------------
    xs = [bin.width  * i / grid_n for i in range(1, grid_n)]
    ys = [bin.height * i / grid_n for i in range(1, grid_n)]

    # 2) 버킷 ------------------------------------------------------------
    w_grid   = np.zeros((grid_n, grid_n), dtype=float)
    moments  = []

    def _add(it, w):
        x, y, _ = it.b_position
        col = sum(x >= t for t in xs)
        row = sum(y >= t for t in ys)

        # 셀 가중치 반영
        w *= cell_weight[row, col]
        w_grid[row, col] += w

        # 모멘트(무게×거리) 누적
        it_w, it_h, _ = it.getDimension()
        dx = (x + it_w/2) - bin.width  * 0.5
        dy = (y + it_h/2) - bin.height * 0.5
        moments.append(w * math.hypot(dx, dy))

    # ── bin 내부 아이템
    for iid in bin.item_ids:
        i = global_item_manager.get(iid)
        if i is not None:
            _add(i, i.weight)

    # ── 후보 아이템
    if cand_item is not None:
        _add(cand_item, cand_item.weight)

    # 3) 표준편차(분산) + 4) 모멘트 --------------------------------------
    flat = w_grid.flatten()
    var  = np.var(flat)                     # σ²
    mom  = sum(moments) / (flat.sum() + 1e-8)

    raw   = alpha * var + beta * mom
    score = (raw + 1e-9) ** gamma          # 작을수록 ↑, γ>1 확대

    return score


def balance_score_old(bin, cand_item=None,
                         alpha=1.0, beta=0.4, gamma=1.2):
    cx0, cy0 = bin.width*0.5, bin.height*0.5
    T_vec = np.zeros(2)      # Σ m r  (x,y)
    I = 0.0                  # ½(Ixx+Iyy)

    def _acc(it):
        nonlocal I, T_vec
        w, h, _ = it.getDimension()
        mx = it.b_position[0] + w*0.5 - cx0
        my = it.b_position[1] + h*0.5 - cy0
        m  = it.weight
        T_vec += m * np.array([mx, my])
        I += 0.5 * m * (mx*mx + my*my)

    for iid in bin.item_ids:
        obj = global_item_manager.get(iid)
        if obj is not None:
            _acc(obj)
    if cand_item is not None:
        _acc(cand_item)

    imbalance = np.linalg.norm(T_vec)         # ||Σ m r||
    score_raw = - (alpha*imbalance - beta*I)
    return (score_raw + 1e-9)**gamma          # ↑클수록 우수

def balance_score(bin, cand_item=None, alpha=1.0, beta=0.4, eps: float = 1e-9):
    """
    질량중심 불균형(imbalance)과 관성모멘트(I)를 동시에 고려한 안정도 점수.
    - imbalance = || Σ m * r ||  (r: bin 중심 기준 2D 오프셋) “질량이 어디로 치우쳐 있는가(=질량중심 오프셋)”를 나타내는 불균형 지표
    - I         = Σ 0.5 * m * ||r||^2
    - 점수      = (β I) / (β I + α imbalance + ε)  ∈ [0,1]
      → I가 클수록(무게가 넓게 퍼질수록) ↑, 불균형이 클수록 ↓

    Parameters
    ----------
    bin : Bin
    cand_item : Item | None
        후보 아이템을 포함한 상태의 점수를 보고 싶다면 넘겨줌.
    alpha, beta : float
        두 항의 상대적 중요도(스케일 가중치)
    eps : float
        0-division 방지용 작은 상수
    """
    W = float(bin.width)
    H = float(bin.height)
    cx0, cy0 = W * 0.5, H * 0.5

    # 0-division 방지
    Wn = W if W > 0 else 1.0
    Hn = H if H > 0 else 1.0
    MW = float(getattr(bin, "max_weight", 0.0) or 0.0)
    MWn = MW if MW > 0 else 1.0

    T_vec = np.zeros(2, dtype=float)  # Σ (m̃ * r̃)
    I_tilde = 0.0                     # Σ 0.5 * m̃ * ||r̃||^2

    def _acc(it):
        nonlocal I_tilde, T_vec
        w, h, _ = it.getDimension()
        # 중심 좌표 (실좌표)
        mx = float(it.b_position[0] + 0.5 * w - cx0)
        my = float(it.b_position[1] + 0.5 * h - cy0)
        m  = float(getattr(it, "weight", 0.0) or 0.0)

        # 무차원화
        rx = mx / Wn
        ry = my / Hn
        m_t = m / MWn

        r2 = rx * rx + ry * ry
        T_vec += m_t * np.array([rx, ry])
        I_tilde += 0.5 * m_t * r2

    # 기존 아이템 누적
    for iid in getattr(bin, "item_ids", []):
        obj = global_item_manager.get(iid)
        if obj is not None:
            _acc(obj)

    # 후보 아이템(옵션)
    if cand_item is not None:
        _acc(cand_item)

    imbalance = float(np.linalg.norm(T_vec))  # ||Σ m̃ r̃||
    num = beta * I_tilde
    den = beta * I_tilde + alpha * imbalance + eps
    score = num / den

    # 수치 안정화
    if score < 0.0: score = 0.0
    if score > 1.0: score = 1.0
    return score

def balance_feature(bin, cand_item=None):
    """
    반환:
      raw_imbalance, raw_I, sum_m_t
    - rx = (x_center - W/2)/W  ∈ [-0.5,0.5]
    - ry = (y_center - H/2)/H  ∈ [-0.5,0.5]
    - m_t = m / max_weight
    - imbalance_raw = || Σ m_t * [rx, ry] ||
    - I_raw         = Σ 0.5 * m_t * (rx^2 + ry^2)
    """
    W = float(bin.width); H = float(bin.height)
    cx0, cy0 = W * 0.5, H * 0.5
    Wn = W if W > 0 else 1.0
    Hn = H if H > 0 else 1.0
    MW = float(getattr(bin, "max_weight", 0.0) or 0.0)
    MWn = MW if MW > 0 else 1.0

    T_vec = np.zeros(2, dtype=float)
    I_tilde = 0.0
    sum_m_t = 0.0

    def _acc(it):
        nonlocal I_tilde, T_vec, sum_m_t
        w, h, _ = it.getDimension()
        mx = float(it.b_position[0] + 0.5 * w - cx0)
        my = float(it.b_position[1] + 0.5 * h - cy0)
        m  = float(getattr(it, "weight", 0.0) or 0.0)

        rx = mx / Wn
        ry = my / Hn
        m_t = m / MWn

        r2 = rx*rx + ry*ry
        T_vec   += m_t * np.array([rx, ry])
        I_tilde += 0.5 * m_t * r2
        sum_m_t += m_t

    # 기존 + 후보
    for iid in getattr(bin, "item_ids", []):
        obj = global_item_manager.get(iid)
        if obj is not None:
            _acc(obj)
    if cand_item is not None:
        _acc(cand_item)

    imbalance_raw = float(np.linalg.norm(T_vec))
    return imbalance_raw, float(I_tilde), float(sum_m_t)

def normalize_balance_features(imb_raw: float, I_raw: float, sum_m_t: float):
    """
    [0,1] 정규화:
      imbalance_norm = imb_raw / (sqrt(0.5) * sum_m_t)
      I_norm         = I_raw / (0.25 * sum_m_t)
    과적재 등으로 sum_m_t>1 가능 → 그대로 상한 계산에 반영.
    sum_m_t≈0이면 0 반환.
    """
    denom_imb = (np.sqrt(0.5) * max(sum_m_t, 1e-9))
    denom_I   = (0.25 * max(sum_m_t, 1e-9))

    imb_norm = float(np.clip(imb_raw / denom_imb, 0.0, 1.0))
    I_norm   = float(np.clip(I_raw   / denom_I,   0.0, 1.0))
    return imb_norm, I_norm

def balance_term_capped(su_term: float, bin_obj, cand_item=None, balance_ratio_cap=0.1) -> float:
    """
    balance_feature / normalize_balance_features 기반으로 안정도를 계산하고,
    SU항 크기 * balance_ratio_cap 을 상한(cap)으로 하는 [-cap, +cap] 범위 값으로 변환.
    """
    imb_raw, I_raw, sum_m_t = balance_feature(bin_obj, cand_item=cand_item)
    imb_n, I_n = normalize_balance_features(imb_raw, I_raw, sum_m_t)  # ∈ [0,1]

    # 균형이 좋을수록 score → 1, 나쁠수록 → 0
    score = 0.5 * ((1.0 - imb_n) + (1.0 - I_n))  # ∈ [0,1]

    cap = balance_ratio_cap * abs(su_term)       # 상한값 (0이면 밸런스 항도 0)
    return cap * (2.0 * score - 1.0)             # [-cap, +cap] 로 매핑