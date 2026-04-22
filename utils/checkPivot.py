
# utils/checkPivot.py
from .overlap import checkOverlap, checkOverlap_with_margin, compute_overlap_area
from .get_value import get_bounds
from .constants import EPS

import math
from typing import Final, Tuple, List
import copy
import numpy as np

# ────────────────── 결과 넘버링 ────────────────────────────
SUCCESS = 1                # 아이템을 적재할 수 있음
# ── 실패 코드 ─────────────────────────────────────────────
FAIL_OUT_OF_BOUNDS_NEG = -1    # 아이템이 bin의 경계를 벗어남
FAIL_OUT_OF_BOUNDS_POS = -2      # 아이템이 bin의 크기를 초과함
FAIL_COLLISION = -3               # 아이템이 bin의 다른 아이템과 충돌
FAIL_WEIGHT_EXCEEDED = -4    # bin의 최대 하중을 초과함
FAIL_NO_TOP_EMPTY = -5          # bin의 상단에 아이템을 둘 공간이 없음
FAIL_NO_SUPPORT_BOTTOM = -6       # 아이템을 지지할 바닥면 아이템
FAIL_SUPPORT_OVERLOAD = -7  # 바닥면 아이템의 하중을 초과함
FAIL_OVERHANG_TOO_MUCH = -8  # 아이템의 겹침 비율이 부족
FAIL_SUPPORT_AREA_INSUFFICIENT = -9  # 필로우(overlap area) 비율이 부족
FAIL_CG_OUTSIDE_SUPPORT = -10  # 아이템의 CG가 지지면의 볼록 껍질 안에 없음
FAIL_CUMULATIVE_UNSTABLE = -11  # 아이템이 누적 지지 조건을 만족하지 않음

def checkPivot_R(bin, item, pivot_pos, rotation_quat, apply_margin=False):
    """
    bin의 pivot 위치에 item을 둘 수 있는지 검사한다.
    반환: (적재 가능 여부 코드, Item 객체 복사본)
    """
    bk_item = copy.deepcopy(item)

    # ────────────────────────────────────────────────
    # 0) 기본 세팅
    pivot_pos = [round(p, 2) for p in pivot_pos]  # 소수점 2자리
    bk_item.b_position = pivot_pos
    bk_item.rotation_quat = rotation_quat
    pivot_ex, pivot_ey, pivot_ez = bk_item.ex, bk_item.ey, bk_item.ez

    # # # 디버깅
    # bin.render(pivots=[pivot_pos, [pivot_ex, pivot_ey, pivot_ez]], show=True)
    # ────────────────────────────────────────────────
    # 1. 경계 검사
    if min(bk_item.b_position) < 0:
        return FAIL_OUT_OF_BOUNDS_NEG, bk_item

    # 2. bin 크기 초과 검사
    if item.name == "gripper":
        if bin.width < pivot_ex or bin.height < pivot_ey:
            return FAIL_OUT_OF_BOUNDS_POS, bk_item
    else:
        if (
            bk_item.b_position[0] < 0 or
            bk_item.b_position[1] < 0 or
            bk_item.b_position[2] < 0 or
            pivot_ex > bin.width or
            pivot_ey > bin.height or
            pivot_ez > bin.depth
        ):
            return FAIL_OUT_OF_BOUNDS_POS, bk_item

    # 3. 충돌 검사 (margin 고려)
    use_margin = (apply_margin and
                  not (np.isclose(pivot_ex, bin.width) or
                       np.isclose(pivot_ey, bin.height)))
    # print(f' -> Checking collision (use_margin={use_margin})...')
    collision = (checkOverlap_with_margin if use_margin else checkOverlap)(bin, bk_item)
    # print(f'    -> collision={collision}')
    if collision:
        return FAIL_COLLISION, bk_item

    # 4. 최대 하중
    if bin.getTotalWeight() + bk_item.weight > bin.max_weight:
        return FAIL_WEIGHT_EXCEEDED, bk_item

    # 5. 상단 경로 확보
    if not bin.is_top_empty(bk_item):
        return FAIL_NO_TOP_EMPTY, bk_item

    # 5.5. 그리퍼 특례
    if bk_item.name == "gripper":
        return SUCCESS, bk_item

    # 6. 바닥면일 경우 바로 통과
    if np.isclose(pivot_pos[2], 0.0):
        bk_item._bottom_overlap_area = 1.0
        return SUCCESS, bk_item

    # ────────────────────────────────────────────────
    # 6. 지지 아이템 필터링
    # if bk_item._id == 8 and pivot_pos == [307.0, 0.0, 104.0]:
    #     print('debug')
    bottom_items = bin.get_bottom_items(bk_item)

    if not bottom_items:
        return FAIL_NO_SUPPORT_BOTTOM, bk_item

    # 7. 지지물 하중 검사
    is_safe, failed_bot = check_load_bearing(bin, bk_item, bottom_items)
    if not is_safe:
        return FAIL_SUPPORT_OVERLOAD, failed_bot

    # ────────────────────────────────────────────────
    # 8. 지지면(오버행/브리지) 검사
    support_pts = []
    bk_item._bottom_overlap_area = 0.0
    x_segs, y_segs = [], []          # 축별 겹침 구간

    ix0, ix1, iy0, iy1, *_ = get_bounds(bk_item.getVertices())
    item_w, item_h = ix1 - ix0, iy1 - iy0

    for bot in bottom_items:
        bx0,bx1, by0,by1, *_ = get_bounds(bot.getVertices())
        x0, x1 = max(bx0, ix0), min(bx1, ix1)
        y0, y1 = max(by0, iy0), min(by1, iy1)
        if x1 <= x0 or y1 <= y0:
            continue

        # 면적 비율 누적
        overlap_area = (x1-x0)*(y1-y0)
        base_area = item_w * item_h
        bk_item._bottom_overlap_area += overlap_area / base_area

        # 축별 구간 저장
        x_segs.append((x0, x1))
        y_segs.append((y0, y1))

        # 지지 점 4개
        support_pts.extend([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])

    # 전체 겹침 길이 계산 유틸
    def union_len(segs):
        if not segs:
            return 0.0
        segs.sort()
        total, s, e = 0.0, *segs[0]
        for ns, ne in segs[1:]:
            if ns > e:
                total += e - s
                s, e = ns, ne
            else:
                e = max(e, ne)
        return total + (e - s)

    tot_ovl_x = union_len(x_segs)
    tot_ovl_y = union_len(y_segs)
    req_ratio = 0.68 if len(bottom_items) == 1 else 0.20

    # 8. 겹침 길이 비율 검사
    if (tot_ovl_x < req_ratio * item_w) or (tot_ovl_y < req_ratio * item_h):
        return FAIL_OVERHANG_TOO_MUCH, bk_item

    # 9. 필로우(overlap area) 비율 체크
    if round(bk_item._bottom_overlap_area, 3) < bin.support_surface_ratio:
        return FAIL_SUPPORT_AREA_INSUFFICIENT, bk_item

    # ────────────────────────────────────────────────
    # 10, 11. 안정성(무게중심 & 누적) 검사
    support_pts = list({pt for pt in support_pts})
    cx = pivot_pos[0] + item_w / 2.0
    cy = pivot_pos[1] + item_h / 2.0

    if not point_in_poly((cx, cy), convex_hull(support_pts)):
        return FAIL_CG_OUTSIDE_SUPPORT, bk_item
    if not is_cumulatively_supported(bin, bk_item, bottom_items):
        return FAIL_CUMULATIVE_UNSTABLE, bk_item

    return SUCCESS, bk_item



# ───────────────── 누적 지지-검사 (다층 CG + 하중) ────────────────
SAFETY_RATIO = 0.9          # ⬅️  원하는 안전율(α) 값 0~1

def is_cumulatively_supported(bin, top_item, first_layer) -> bool:
    """
    다층 브리지(bridge)는 허용, 다층 캔틸레버(cantilever)는 거부.
    ─ 검사 로직 ──────────────────────────────────────────────
    ① 위→아래로 내려가며, 위에 쌓인 *전체 스택*을 하나의
       rigid body(질량 cum_W, CG cum_C) 로 본다.
    ② 현재 층 아이템들의 Convex-Hull 안에 CG 가 들어 있는지 확인.
    ③ 그 질량 cum_W 가 현재 층 모든 아이템의 loadbear 를 초과하지
       않는지 확인.
    두 조건을 한 층이라도 만족하지 못하면 False.
    """

    # 0) 층집합 모으기 (위에서 아래로)
    layers   = [first_layer]
    visited  = {i._id for i in first_layer}
    cur = first_layer
    while cur:
        nxt = []
        for it in cur:
            for b in bin.get_bottom_items_in_graph(it):
                if b._id not in visited:
                    visited.add(b._id)
                    nxt.append(b)
        if not nxt:
            break
        layers.append(nxt)
        cur = nxt

    # 1) 초기 누적 질량·CG  = top_item 하나
    cum_W = top_item.weight
    w, h, _ = top_item.getDimension()
    cum_Cx = top_item.b_position[0] + w/2
    cum_Cy = top_item.b_position[1] + h/2

    # 2) 층별 검사
    for bots in layers:
        # (a) CG ⇢ 지지 다각형 내?
        pts = []
        for b in bots:
            bx0,bx1, by0,by1, *_ = get_bounds(b.getVertices())
            pts += [(bx0,by0), (bx1,by0), (bx1,by1), (bx0,by1)]

        hull = convex_hull(pts)

        # ★ 1)  hull 안쪽으로 α 배 축소
        if SAFETY_RATIO < 1.0:
            cx_h, cy_h = polygon_centroid(hull)
            hull = [(
                cx_h + (x - cx_h) * SAFETY_RATIO,
                cy_h + (y - cy_h) * SAFETY_RATIO
            ) for x, y in hull]

        if not point_in_poly((cum_Cx, cum_Cy), hull):
            return False                            # ★ 캔틸레버 NG

        # (b) 하중 ⇢ loadbear 이내?
        for b in bots:
            if cum_W > b.loadbear + EPS:
                return False                        # ★ 하중 초과 NG

        # (c) 누적 질량·CG 갱신 (다음 층용)
        layer_W  = sum(b.weight for b in bots)
        if layer_W:
            # 각 아이템별 footprint 중심 사용
            layer_Cx = sum(b.weight * (b.b_position[0] + b.getDimension()[0]/2)
                           for b in bots) / layer_W
            layer_Cy = sum(b.weight * (b.b_position[1] + b.getDimension()[1]/2)
                           for b in bots) / layer_W
            cum_Cx = (cum_Cx*cum_W + layer_Cx*layer_W) / (cum_W + layer_W)
            cum_Cy = (cum_Cy*cum_W + layer_Cy*layer_W) / (cum_W + layer_W)
            cum_W  += layer_W

    return True      # 모든 층 통과 ⇒ 안정

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull(pts: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    pts = sorted(set(pts))
    if len(pts) <= 1:
        return pts
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def point_in_poly(pt, poly):
    # ray casting O(n)
    x, y = pt
    inside = False
    for i in range(len(poly)):
        x0,y0 = poly[i]
        x1,y1 = poly[(i+1)%len(poly)]
        if ((y0 > y) != (y1 > y)) and \
           (x < (x1-x0)*(y-y0)/(y1-y0+1e-12) + x0):
            inside = not inside
    return inside

def polygon_centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    입력:   pts = [(x0,y0), (x1,y1), ...]  ← 첫점과 끝점은 동일하지 않아도 됨
    출력:   (Cx, Cy)  : 다각형의 면적 중심
    관광:   Shoelace + 1/6A Σ(x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
    주의:   꼭짓점이 3개 미만(면적 0) → (평균 x, 평균 y) 반환
    """
    n = len(pts)
    if n < 3:
        xs, ys = zip(*pts)
        return sum(xs) / n, sum(ys) / n

    A = 0.0
    Cx = 0.0
    Cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0     # = 2 * sub-triangle area (signed)
        A  += cross
        Cx += (x0 + x1) * cross
        Cy += (y0 + y1) * cross

    A *= 0.5
    if math.isclose(A, 0.0, abs_tol=1e-9):
        # 면적 0 → 단순 평균
        xs, ys = zip(*pts)
        return sum(xs) / n, sum(ys) / n

    Cx /= (6.0 * A)
    Cy /= (6.0 * A)
    return Cx, Cy


def get_item_bounds_tuple(item):
    """
    compute_overlap_area 함수가 요구하는 6-tuple 형식으로 변환
    (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    return (
        item.b_position[0], item.ex,
        item.b_position[1], item.ey,
        item.b_position[2], item.ez
    )

def check_load_bearing(bin, bk_item, bottom_items):
    """
    면적 비율(Overlap Ratio)을 적용한 정밀 하중 검사
    """
    # 1. 현재 배치하려는 아이템(bk_item)의 바닥 면적 및 Bounds
    bk_bounds = get_item_bounds_tuple(bk_item)
    bk_item_w, bk_item_h, _ = bk_item.getDimension()
    bk_base_area = bk_item_w * bk_item_h
    
    # 0으로 나누기 방지
    if bk_base_area <= EPS:
        return True # 혹은 에러 처리

    # 2. 지지물(bottom_items) 순회
    for bot in bottom_items:
        bot_bounds = get_item_bounds_tuple(bot)

        # ---------------------------------------------------------
        # [Step A] 새 아이템(bk_item)이 bot에게 주는 하중 계산
        # ---------------------------------------------------------
        # overlap.py의 함수 사용하여 겹침 면적 계산
        overlap_area = compute_overlap_area(bot_bounds, bk_bounds)

        # 겹치지 않으면 하중 전달 없음
        if overlap_area <= EPS:
            continue

        # 하중 비율: (겹친 면적 / bk_item의 전체 바닥 면적)
        load_ratio = overlap_area / bk_base_area
        added_weight = bk_item_w * load_ratio

        # ---------------------------------------------------------
        # [Step B] bot이 '이미' 견디고 있는 하중 계산 (면적 비율 적용)
        # ---------------------------------------------------------
        current_load_on_bot = 0.0
        
        # bot 위에 있는 기존 아이템들을 가져옴
        top_of_bot = bin.get_items_above(bot) 
        
        for top_item in top_of_bot:
            top_bounds = get_item_bounds_tuple(top_item)
            top_item_w, top_item_h, _ = top_item.getDimension()
            top_base_area = top_item_w * top_item_h
            
            # bot과 기존 상단 아이템 간의 겹침 면적
            existing_overlap = compute_overlap_area(bot_bounds, top_bounds)
            
            if existing_overlap > EPS and top_base_area > EPS:
                # 기존 아이템이 bot에게 가하는 하중 누적
                existing_ratio = existing_overlap / top_base_area
                current_load_on_bot += top_item_w * existing_ratio

        # ---------------------------------------------------------
        # [Step C] 최종 하중 판정
        # ---------------------------------------------------------
        # (기존 하중 + 새로 추가될 하중) > 지지 하중 한계(loadbear)
        if current_load_on_bot + added_weight > bot.loadbear:
            # 실패 시 로그를 남기거나 특정 상수를 반환
            return False, bot  # (실패 코드, 실패한 지지물)

    # 모든 지지물에 대해 검사 통과
    return True, None