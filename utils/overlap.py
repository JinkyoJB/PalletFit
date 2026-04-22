# utils/overlap.py
from __future__ import annotations

import numpy as np


from planning.itemManager import global_item_manager
from utils.constants import EPS
from typing import Final, Tuple

# ──────────────────────────────────────────────────────
# 1. overlap 헬퍼
# ──────────────────────────────────────────────────────
def overlap_intervals(
    a1: float, a2: float, b1: float, b2: float,
    *, include_touch: bool = False
) -> bool:
    """
    두 구간이 실제로 *겹치면* True.
    include_touch=False → 끝점만 닿으면 False.
    """
    gap = min(a2, b2) - max(a1, b1)   # (+)겹침, 0 맞닿음, (-)떨어짐

    if include_touch:          # 닿기만 해도 OK
        return gap >= -EPS     # 수치 오차 허용
    else:                      # 닿으면 겹침 아님
        return gap >  EPS      # 반드시 '양수'여야 겹침


def overlap_length(a1: float, a2: float,b1: float, b2: float) -> float:
    """
    두 구간 [a1,a2], [b1,b2] 가 겹치는 길이(음수면 0).
    """
    if not overlap_intervals(a1, a2, b1, b2):
        return 0.0
    return min(a2, b2) - max(a1, b1)

def overlap_volume(boxA, boxB) -> float:
    (Ax0,Ay0,Az0), (Ax1,Ay1,Az1) = boxA
    (Bx0,By0,Bz0), (Bx1,By1,Bz1) = boxB

    ox = overlap_length(Ax0, Ax1, Bx0, Bx1)
    oy = overlap_length(Ay0, Ay1, By0, By1)
    oz = overlap_length(Az0, Az1, Bz0, Bz1)

    return ox * oy * oz        # 셋 중 하나라도 0이면 부피는 0


# ──────────────────────────────────────────────────────
# 1-1. overlap 헬퍼 활용 함수
# ──────────────────────────────────────────────────────

def _expand_bounds(xmin: float, xmax: float, margin: float) -> tuple[float, float]:
    """양쪽으로 margin/2 만큼 확장"""
    half = margin * 0.5
    return xmin - half, xmax + half


def _shrink_bounds(zmin: float, zmax: float, margin: float) -> tuple[float, float]:
    """
    z‑축은 margin/2 만큼 ‘안쪽으로’ 잘라낸다.
    (겹침을 더 엄격하게 본다)
    """
    half = margin * 0.5
    zmin += half
    zmax -= half
    # 만약 interval 이 뒤집히면 한 점으로 처리
    if zmax < zmin:
        mid = 0.5 * (zmin + zmax)
        zmin = zmax = mid
    return zmin, zmax

def overlap_item_aabb_with_margin(
    itemA,
    itemB,
    margin_x: float,
    margin_y: float,
    margin_z: float = 2.5,          # z‑축 margin (수축)
) -> bool:
    """
    x·y 축은 margin 을 ‘확대’, z 축은 margin 을 ‘수축’하여
    두 AABB 가 겹치는지 판정한다.
    """
    # ── 원본 경계값 --------------------------------------------------
    Ax0, Ay0, Az0 = itemA.b_position
    Ax1, Ay1, Az1 = itemA.ex, itemA.ey, itemA.ez
    Bx0, By0, Bz0 = itemB.b_position
    Bx1, By1, Bz1 = itemB.ex, itemB.ey, itemB.ez

    # ── margin 반영 --------------------------------------------------
    Ax0, Ax1 = _expand_bounds(Ax0, Ax1, margin_x)
    Ay0, Ay1 = _expand_bounds(Ay0, Ay1, margin_y)
    Az0, Az1 = _shrink_bounds(Az0, Az1, margin_z)

    Bx0, Bx1 = _expand_bounds(Bx0, Bx1, margin_x)
    By0, By1 = _expand_bounds(By0, By1, margin_y)
    Bz0, Bz1 = _shrink_bounds(Bz0, Bz1, margin_z)

    # ── 모든 축이 겹치면 충돌 ---------------------------------------
    return (
        Ax0 < Bx1 and Ax1 > Bx0 and   # x‑축 (확장)
        Ay0 < By1 and Ay1 > By0 and   # y‑축 (확장)
        Az0 < Bz1 and Az1 > Bz0       # z‑축 (수축)
    )

def compute_overlap_area(pivot_bounds, item_bounds):
    """
    두 직육면체의 x-y 평면에서의 겹치는 면적(mm²)을 반환
    pivot_bounds, item_bounds = (x_min,x_max,y_min,y_max,z_min,z_max)
    """
    px_min, px_max, py_min, py_max, *_ = pivot_bounds
    ix_min, ix_max, iy_min, iy_max, *_ = item_bounds

    overlap_x = overlap_length(px_min, px_max, ix_min, ix_max)
    overlap_y = overlap_length(py_min, py_max, iy_min, iy_max)

    return overlap_x * overlap_y          # 겹치는 면적

def compute_overlap_volume(box1, box2, ) -> float:
    """
    box1, box2 = (xmin,xmax,ymin,ymax,zmin,zmax)
    두 AABB 의 교차 부피를 반환. (겹치지 않으면 0)
    """
    return overlap_volume(
        _flat_to_box(box1),
        _flat_to_box(box2),
    )

def overlap_item_aabb(itemA, itemB) -> bool:
    Ax_min, Ay_min, Az_min = itemA.b_position
    Ax_max, Ay_max, Az_max = itemA.ex, itemA.ey, itemA.ez
    Bx_min, By_min, Bz_min = itemB.b_position
    Bx_max, By_max, Bz_max = itemB.ex, itemB.ey, itemB.ez

    return (
        overlap_intervals(Ax_min, Ax_max, Bx_min, Bx_max, include_touch=False) and
        overlap_intervals(Ay_min, Ay_max, By_min, By_max, include_touch=False) and
        overlap_intervals(Az_min, Az_max, Bz_min, Bz_max, include_touch=False)
    )

def overlap_item(itemA, itemB):
    """
    두 Item 객체가 겹치는지 판별하는 함수.
    """
    boxA = item_to_box(itemA)
    boxB = item_to_box(itemB)
    return boxes_intersect_3d(boxA, boxB)


# ──────────────────────────────────────────────────────
# 2. 변환 헬퍼
# ──────────────────────────────────────────────────────
def _flat_to_box(bounds: Tuple[float,float,float,float,float,float]):
    """(xmin,xmax,ymin,ymax,zmin,zmax) → [[xmin,ymin,zmin],[xmax,ymax,zmax]]"""
    x_min,x_max,y_min,y_max,z_min,z_max = bounds
    return [[x_min, y_min, z_min], [x_max, y_max, z_max]]

def item_to_box(item):
    """
    Item 객체 -> AABB
    box = [ [x_min, y_min, z_min], [x_max, y_max, z_max] ]
    """
    x_min, y_min, z_min = item.b_position
    x_max, y_max, z_max = item.ex, item.ey, item.ez
    return [[x_min, y_min, z_min], [x_max, y_max, z_max]]

def get_box_vertices(box):
    """
    box = [ [x_min, y_min, z_min], [x_max, y_max, z_max] ]
    8개의 꼭짓점(정점)을 반환
    순서는 편한 대로 하되 일관성만 지키면 됨
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = box
    
    return [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]

def get_box_faces_vertices(box):
    """
    box를 구성하는 6개 면(각각 사각형)의 꼭짓점 리스트(4점짜리)를 반환.
    [
      [p0, p1, p2, p3],  # 아래
      [p4, p5, p6, p7],  # 위
      [p0, p1, p5, p4],  # 앞
      [p3, p2, p6, p7],  # 뒤
      [p0, p3, p7, p4],  # 왼
      [p1, p2, p6, p5],  # 오른
    ]
    """
    verts = get_box_vertices(box)  # 8개 정점
    # 인덱스 매핑
    faces_idx = [
        (0, 1, 2, 3),  # 아래
        (4, 5, 6, 7),  # 위
        (0, 1, 5, 4),  # 앞
        (3, 2, 6, 7),  # 뒤
        (0, 3, 7, 4),  # 왼
        (1, 2, 6, 5),  # 오른
    ]
    faces = []
    for f in faces_idx:
        quad = [verts[i] for i in f]
        faces.append(quad)
    return faces

# ──────────────────────────────────────────────────────
# 3. 충돌 검사 관련 함수
# ──────────────────────────────────────────────────────
def checkOverlap(bin, item):
    """
    bin.binIndex를 사용해 빠르게 후보군을 좁힌 후,
    최종적으로 정확한 AABB 충돌 판정(overlap_item_aabb)을 수행한다.
    """

    # 1. b_position 기준으로 빠르게 후보군 좁히기
    xy_candidates = bin.search_xy(item.b_position[0], item.ex, item.b_position[1], item.ey)
    xz_candidates = bin.search_xz(item.b_position[0], item.ex, item.b_position[2], item.ez)
    yz_candidates = bin.search_yz(item.b_position[1], item.ey, item.b_position[2], item.ez)
    
    # 2. 3개의 후보군에 모두 있는 아이템만 남기기
    search_list = list(set(xy_candidates) & set(xz_candidates) & set(yz_candidates))

    # 아이템 리스트로 반환
    search_list = [global_item_manager.get(item_id) for item_id in search_list]
    # search_list = bin.get_all_items()

    # 3. 최종적으로 AABB 충돌 판정
    return _checkOverlap(search_list, item)

def _checkOverlap(search_list, item):
    """
    현재 트리에 들어있는 모든 아이템들과, 
    인자로 주어진 'item'이 3D 축 정렬 박스 기준으로 전혀 겹치지 않는지 판별합니다.
    """
    for other_item in search_list:
        # if other_item is item:
        #     # 만약 'item'이 이미 트리에 들어있는 동일 객체라면
        #     # 자기 자신과의 비교는 스킵할 수도 있음
        #     continue
        if item.name == 'gripper' and other_item.options.get('is_attached') is True:
            # 그리퍼와 타겟 아이템의 충돌은 무시
            continue

        if overlap_item_aabb(item, other_item):
            return True
    return False

def checkOverlap_with_margin(bin, item):
    """
    bin.binIndex를 사용해 빠르게 후보군을 좁힌 후,
    최종적으로 정확한 AABB 충돌 판정(overlap_item_aabb)을 수행한다.
    """
    # 1. b_position 기준으로 빠르게 후보군 좁히기
    x_min = item.b_position[0]
    x_max = item.ex
    y_min = item.b_position[1]
    y_max = item.ey
    z_min = item.b_position[2]
    z_max = item.ez

    # if item._id==12 and item.b_position[0]==0.0 and item.b_position[1]==87.0 and item.b_position[2]==104.0:
    #     print('debug')

    xy_candidates = bin.search_xy(x_min, x_max, y_min, y_max)
    xz_candidates = bin.search_xz(x_min, x_max, z_min, z_max)
    yz_candidates = bin.search_yz(y_min, y_max, z_min, z_max)

    # 2. 3개의 후보군에 모두 있는 아이템만 남기기
    search_list = list(set(xy_candidates) | set(xz_candidates) | set(yz_candidates))

    # 아이템 리스트로 반환
    search_list = [global_item_manager.get(item_id) for item_id in search_list]
    # search_list = bin.get_all_items()


    # 3. 최종적으로 AABB 충돌 판정
    return _checkOverlap_with_margin(search_list, item, bin)

def _checkOverlap_with_margin(search_list, item, bin):
    """
    bruth-force 방식으로,
    현재 트리에 들어있는 모든 아이템들과, 
    인자로 주어진 'item'이 3D 축 정렬 박스 기준으로 전혀 겹치지 않는지 판별합니다.
    """
    for other_item in search_list:
        # if other_item is item:
        #     # 만약 'item'이 이미 트리에 들어있는 동일 객체라면
        #     # 자기 자신과의 비교는 스킵할 수도 있음
        #     continue

        if item.name == 'gripper' and other_item.options['is_attached'] is True:
            # 그리퍼와 타겟 아이템의 충돌은 무시
            continue

        if overlap_item_aabb_with_margin(item, other_item,
                                         margin_x=bin.margin_x,
                                         margin_y=bin.margin_y):
            # item과 other_item이 겹치면 True
            return True
    return False



# ──────────────────────────────────────────────────────
# 4. 삼각형 충돌 검사
# ──────────────────────────────────────────────────────
def boxes_intersect_3d(boxA, boxB):
    """
    boxA, boxB = [ [x_min, y_min, z_min], [x_max, y_max, z_max] ]
    
    1) boxA의 각 face(6개 사각형) vs boxB의 각 face(6개 사각형)
       => rectangle_intersect() 로 검사
    2) 어떤 면이라도 교차하면 True
    3) 면 교차가 없다면, boxA의 꼭짓점 중 하나가 boxB 내부인지
       또는 boxB의 꼭짓점 중 하나가 boxA 내부인지 검사
    4) 둘 중 하나라도 내부면 True, 아니면 False
    """
    facesA = get_box_faces_vertices(boxA)  # 6개 사각형
    facesB = get_box_faces_vertices(boxB)  # 6개 사각형

    # (1) 면-면 교차 검사
    for quadA in facesA:
        for quadB in facesB:
            if rectangle_intersect(quadA, quadB):
                return True
    
    # (2) 면 교차가 전혀 없다면 => 내부 포함 여부 검사
    vertsA = get_box_vertices(boxA)
    vertsB = get_box_vertices(boxB)

    # boxA의 첫 꼭짓점이 boxB 내부인가?
    if point_in_box(vertsA[0], boxB):
        return True
    # boxB의 첫 꼭짓점이 boxA 내부인가?
    if point_in_box(vertsB[0], boxA):
        return True

    return False

def segments_intersect(seg1, seg2, epsilon=1e-6):
    A, B = np.array(seg1[0]), np.array(seg1[1])
    C, D = np.array(seg2[0]), np.array(seg2[1])
    
    u = B - A
    v = D - C
    w = A - C
    
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    
    denom = a * c - b * b
    if abs(denom) < epsilon:
        # 선분이 거의 평행함
        return False
    
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    # 선분 범위 안에 존재하는지 확인
    if 0 <= s <= 1 and 0 <= t <= 1:
        P = A + s * u
        Q = C + t * v
        # 교점이 거의 일치하면 교차한다고 판단
        return np.linalg.norm(P - Q) < epsilon
    else:
        return False
    
def triangle_intersect(V0, V1, V2, U0, U1, U2):
    """Möller’s triangle-triangle intersection test"""
    # Bounding box check first (for quick rejection)
    V = np.array([V0, V1, V2])
    U = np.array([U0, U1, U2])
    if (np.max(V, axis=0) < np.min(U, axis=0)).any() or (np.max(U, axis=0) < np.min(V, axis=0)).any():
        return False

    # Use a robust library or implement full triangle-triangle intersection test
    # Here, simplified: check edge-edge intersections
    for i in range(3):
        for j in range(3):
            p1 = V[i]
            p2 = V[(i + 1) % 3]
            q1 = U[j]
            q2 = U[(j + 1) % 3]
            if segments_intersect([p1, p2], [q1, q2]):
                return True

    return False

def rectangle_intersect(quad1, quad2):
    # 쿼드 1 → 삼각형 두 개
    t1a = [quad1[0], quad1[1], quad1[2]]
    t1b = [quad1[0], quad1[2], quad1[3]]
    
    # 쿼드 2 → 삼각형 두 개
    t2a = [quad2[0], quad2[1], quad2[2]]
    t2b = [quad2[0], quad2[2], quad2[3]]
    
    # 삼각형 4쌍에 대해 교차 여부 확인
    tri_pairs = [
        (t1a, t2a),
        (t1a, t2b),
        (t1b, t2a),
        (t1b, t2b),
    ]
    
    for tri1, tri2 in tri_pairs:
        if triangle_intersect(*tri1, *tri2):
            return True
    return False

def point_in_box(pt, box):
    """
    주어진 점(pt)이 AABB 박스(box) 내부에 있는지 검사.
    box = [ [x_min, y_min, z_min], [x_max, y_max, z_max] ]
    pt = [x, y, z]

    return: True if pt is inside or on boundary, False otherwise.
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = box
    x, y, z = pt

    # 경계를 포함하려면 <= / >=
    if x < x_min or x > x_max:
        return False
    if y < y_min or y > y_max:
        return False
    if z < z_min or z > z_max:
        return False
    return True

def aabb_bounds_xyz(item):
    """
    item의 현재 위치/회전을 반영한 AABB bounds 반환
    returns: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    v = np.asarray(item.getVertices(), dtype=np.float64)  # (8,3) or fallback
    x_min, y_min, z_min = v.min(axis=0)
    x_max, y_max, z_max = v.max(axis=0)
    return (float(x_min), float(x_max), float(y_min), float(y_max), float(z_min), float(z_max))
