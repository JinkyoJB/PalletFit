import numpy as np
import math
from typing import Optional, Sequence, Tuple, List



def change_to_predict_coor(item_position, item_scales):
    '''
    predict를 위해 아이템 하단 모서리 좌표로 변환
    ''' 
    # item_scales를 float로 변환
    item_position = [float(item_position[0]), float(item_position[1]), float(item_position[2])]
    item_scales = [float(item_scales[0]), float(item_scales[1]), float(item_scales[2])]
    transform = np.array([
        [1, 0, 0, -item_scales[0]/2],
        [0, 1, 0, -item_scales[1]/2],
        [0, 0, 1, -item_scales[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 아이템 위치를 확장된 형태로 변환 (동차 좌표)
    item_position_homogeneous = np.append(item_position, 1)

    # 변환 적용
    predicted_coor_homogeneous = transform @ item_position_homogeneous

    # 동차 좌표에서 일반 좌표로 변환
    predicted_coor = predicted_coor_homogeneous[:3]
    return predicted_coor

def shift_bin_w2b_position(bin, item, height_offset=6.9):
    '''
    bin의 세계 좌표계(bin.w_position)를 기준으로 변환하여
    bin 내부 좌표계에서의 item_pick_coor를 반환합니다.
    '''
    bwx, bwy, bwz = bin.w_position[:3]  # bin의 월드 좌표계, m단위
    wx, wy, wz = item.w_position[:3]  # 아이템의 월드 좌표계, m단위

    # 아이템의 world좌표계에서 bin의 world좌표계를 빼서 가상좌표계로 변환
    ix = wx - bwx
    iy = wy - bwy
    iz = wz - bwz

    #m-> mm단위로 변환
    ix = ix * 1000
    iy = iy * 1000
    iz = iz * 1000

    # mm->m단위로 변환
    dimension =item.getDimension()   # 아이템의 가로, 세로, 높이

    b_position = change_to_predict_coor([ix, iy, iz], dimension)
    
    # height_offset만큼 z축을 뺌
    b_position[2] -= height_offset
    
    # 2자리 반올림
    b_position = [round(b, 2) for b in b_position]
    if -1 <= b_position[2] <= 1:
        b_position[2] = 0
    return b_position


def rectIntersect(item1, item2, x, y):
    d1 = item1.getDimension()
    d2 = item2.getDimension()

    cx1 = item1.b_position[x] + d1[x]/2
    cy1 = item1.b_position[y] + d1[y]/2
    cx2 = item2.b_position[x] + d2[x]/2
    cy2 = item2.b_position[y] + d2[y]/2

    ix = max(cx1, cx2) - min(cx1, cx2)
    iy = max(cy1, cy2) - min(cy1, cy2)

    return ix < (d1[x]+d2[x])/2 and iy < (d1[y]+d2[y])/2


def rectIntersect_not_rotation(item1, item2, x, y):
    d1 = item1.getDimension_not_rotation()
    d2 = item2.getDimension_not_rotation()

    cx1 = item1.b_position[x] + d1[x]/2
    cy1 = item1.b_position[y] + d1[y]/2
    cx2 = item2.b_position[x] + d2[x]/2
    cy2 = item2.b_position[y] + d2[y]/2

    ix = max(cx1, cx2) - min(cx1, cx2)
    iy = max(cy1, cy2) - min(cy1, cy2)

    return ix < (d1[x]+d2[x])/2 and iy < (d1[y]+d2[y])/2


def intersect(item1, item2):
    return (
        rectIntersect(item1, item2, 0, 1) and
        rectIntersect(item1, item2, 1, 2) and
        rectIntersect(item1, item2, 0, 2)
    )


def Item_b2w_position(item, place_bin):
    '''
    item을 input으로 넣으면, 해당 b_position을 w_position으로 변환하여 반환
    '''
    if item is None or isinstance(item, bool):
        return None
    
    if item.b_position == [-1,-1,-1]:
        return None
    
    item_WHD = item.getDimension()

    # mm -> m
    b_position = mm2m(item.b_position)
    item_WHD = mm2m(item_WHD)
    current_bin_WHD = [place_bin.width, place_bin.height, place_bin.depth]
    current_bin_WHD = mm2m(current_bin_WHD)

    item_pick_coor = change_to_pick_coor(b_position, item_WHD)
    world_position = shift_bin_coor2world(place_bin, item_pick_coor,item.rotation_quat)

    return world_position

def change_to_pick_coor(position, item_scales):
    '''
    picking을 위해 아이템 상단 중앙 좌표로 변환
    '''    
    #b_position을 float로 변환
    position = [float(position[0]), float(position[1]), float(position[2])]
    item_scales = [float(item_scales[0]), float(item_scales[1]), float(item_scales[2])]

    inverse_transform = np.array([
        [1, 0, 0, item_scales[0]/2],  # X축으로 반 스케일만큼 이동
        [0, 1, 0, item_scales[1]/2],  # Y축으로 반 스케일만큼 이동
        [0, 0, 1, item_scales[2]],    # Z축으로 전체 스케일만큼 이동 (상단 중앙)
        [0.0, 0.0, 0.0, 1.0]                  # 동차 좌표
    ])
    

    # 아이템 위치를 확장된 형태로 변환 (동차 좌표)
    predict_position_homogeneous = np.append(position, 1)
    
    # 변환 적용
    pick_coor_homogeneous = inverse_transform @ predict_position_homogeneous

    # 동차 좌표에서 일반 좌표로 변환
    pick_coor = pick_coor_homogeneous[:3]
    return pick_coor

# -----------------ZoneFit 필요한 함수-----------------
def faces_coincide_type(face_bin, face_zone):
    """
    두 face (face_bin, face_zone)가 어느 정도 겹치는지 판단하여
      - "FULL": 완전히 동일한 면 (고정 축 동일 + 나머지 2축 bounds가 완전히 일치)
      - "PARTIAL": 일부만 겹침 (고정 축 동일 + 나머지 2축 bounds가 부분 겹침)
      - None: 겹치지 않음
    face는 (plane, bounds) 튜플이며, bounds는 {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)} 입니다.
    """
    locked_dim_bin, locked_val_bin = get_locked_dim(face_bin)
    locked_dim_zone, locked_val_zone = get_locked_dim(face_zone)

    if locked_dim_bin is None or locked_dim_zone is None:
        return None
    if locked_dim_bin != locked_dim_zone:
        return None
    if locked_val_bin != locked_val_zone:
        return None

    dims = ['x', 'y', 'z']
    other_dims = [d for d in dims if d != locked_dim_bin]

    full_match_count = 0
    for d in other_dims:
        min_bin, max_bin = face_bin[1][d]
        min_zone, max_zone = face_zone[1][d]
        overlap_min = max(min_bin, min_zone)
        overlap_max = min(max_bin, max_zone)
        if overlap_max <= overlap_min:
            return None
        if min_bin == min_zone and max_bin == max_zone:
            full_match_count += 1

    if full_match_count == len(other_dims):
        return "FULL"
    else:
        return "PARTIAL"

def find_coincident_faces(bin_faces, zone_faces):
    """
    bin_faces와 zone_faces를 모두 비교하여, 겹치는 면을 FULL과 PARTIAL로 분류하여 반환합니다.
    반환 형식:
      {
         "full": [(i, j), ...],
         "partial": [(i, j), ...]
      }
    """
    full_list = []
    partial_list = []

    for i, b_face in enumerate(bin_faces):
        for j, z_face in enumerate(zone_faces):
            result = faces_coincide_type(b_face, z_face)
            if result == "FULL":
                full_list.append((i, j))
            elif result == "PARTIAL":
                partial_list.append((i, j))
    return {"full": full_list, "partial": partial_list}

def faces_coincide_type_no_z(face_bin, face_zone):
    """
    z축 고정(위/아래 면)은 무시하고, 나머지 면끼리 겹침을 판별합니다.
    (즉, 'x'나 'y'축 고정된 면끼리)
    """
    locked_dim_bin, locked_val_bin = get_locked_dim(face_bin)
    locked_dim_zone, locked_val_zone = get_locked_dim(face_zone)

    if locked_dim_bin == 'z' or locked_dim_zone == 'z':
        return None

    if locked_dim_bin is None or locked_dim_zone is None:
        return None
    if locked_dim_bin != locked_dim_zone:
        return None
    if locked_val_bin != locked_val_zone:
        return None

    dims = ['x', 'y', 'z']
    other_dims = [d for d in dims if d != locked_dim_bin]

    full_match_count = 0
    for d in other_dims:
        min_bin, max_bin = face_bin[1][d]
        min_zone, max_zone = face_zone[1][d]
        overlap_min = max(min_bin, min_zone)
        overlap_max = min(max_bin, max_zone)
        if overlap_max <= overlap_min:
            return None
        if min_bin == min_zone and max_bin == max_zone:
            full_match_count += 1

    if full_match_count == len(other_dims):
        return "FULL"
    else:
        return "PARTIAL"
    
def find_coincident_faces_without(bin_faces, zone_faces):
    """
    bin_faces와 zone_faces 중에서, z축 고정(위/아래) 면은 무시하고
    x·y축 고정된 면끼리 겹치는 면들을 FULL과 PARTIAL로 분류하여 반환합니다.
    반환 형식은 find_coincident_faces와 동일합니다.
    """
    full_list = []
    partial_list = []

    for i, b_face in enumerate(bin_faces):
        for j, z_face in enumerate(zone_faces):
            result = faces_coincide_type_no_z(b_face, z_face)
            if result == "FULL":
                full_list.append((i, j))
            elif result == "PARTIAL":
                partial_list.append((i, j))
    return {"full": full_list, "partial": partial_list}

def faces_coincide(face_bin, face_zone):
    """
    두 face가 겹치는지 여부를 True/False로 반환합니다.
    조건:
      1. 둘 다 같은 축에 고정되어 있어야 하며 (예: x=0 면)
      2. 고정된 값이 동일해야 하고
      3. 나머지 두 축의 bound가 서로 겹쳐야 함.
    """
    locked_dim_bin, locked_val_bin = get_locked_dim(face_bin)
    locked_dim_zone, locked_val_zone = get_locked_dim(face_zone)

    if locked_dim_bin is None or locked_dim_zone is None:
        return False
    if locked_dim_bin != locked_dim_zone:
        return False
    if locked_val_bin != locked_val_zone:
        return False

    dims = ['x', 'y', 'z']
    other_dims = [d for d in dims if d != locked_dim_bin]

    for d in other_dims:
        min_bin, max_bin = face_bin[1][d]
        min_zone, max_zone = face_zone[1][d]
        overlap_length = min(max_bin, max_zone) - max(min_bin, min_zone)
        if overlap_length <= 0:
            return False
    return True
    
def get_locked_dim(face):
    """
    face 정보는 두 가지 형태를 가질 수 있습니다.
      1. (plane, bounds) 튜플 형태
      2. {'plane': ..., 'bounds': ...} 딕셔너리 형태

    bounds는 {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)} 형태입니다.
    만약 한 축에 대해 min == max이면 그 축이 고정된 것으로 보고 (axis, value)를 반환합니다.
    예: x_min == x_max → ('x', x_min)
         고정 축이 없으면 (None, None) 반환.
    """
    # face가 dict이면, bounds를 face['bounds']로 취함
    if isinstance(face, dict):
        bounds = face.get('bounds', None)
    # face가 튜플이나 리스트이면, 두 번째 요소로부터 bounds를 추출
    elif isinstance(face, (tuple, list)) and len(face) >= 2:
        bounds = face[1]
    else:
        raise ValueError("Unexpected face structure.")

    if not isinstance(bounds, dict):
        raise ValueError("Expected bounds to be a dictionary, but got: {}".format(type(bounds)))

    x_min, x_max = bounds.get('x', (None, None))
    y_min, y_max = bounds.get('y', (None, None))
    z_min, z_max = bounds.get('z', (None, None))

    if x_min is None or y_min is None or z_min is None:
        raise ValueError("Bounds dictionary is missing required keys.")

    if x_min == x_max:
        return ('x', x_min)
    elif y_min == y_max:
        return ('y', y_min)
    elif z_min == z_max:
        return ('z', z_min)
    else:
        return (None, None)


def find_coincident_faces_list(bin_faces, zone_faces):
    """
    bin_faces와 zone_faces(모두 딕셔너리 형태)를 비교하여, 겹치는 면의 (bin_face_key, zone_face_key) 쌍을 반환합니다.
    단, z축 고정(즉, 'top'과 'bottom') 면은 검사하지 않습니다.
    """
    coincident_pairs = []

    for bin_key, b_face in bin_faces.items():
        # z축 고정 면은 건너뛰기 ('top'과 'bottom')
        if bin_key in ['top', 'bottom']:
            continue
        locked_dim_bin, _ = get_locked_dim(b_face)
        if locked_dim_bin == 'z':
            continue

        for zone_key, z_face in zone_faces.items():
            if zone_key in ['top', 'bottom']:
                continue
            locked_dim_zone, _ = get_locked_dim(z_face)
            if locked_dim_zone == 'z':
                continue

            if faces_coincide(b_face, z_face):
                coincident_pairs.append((bin_key, zone_key))
    
    return coincident_pairs

# ----------------- OutLineFit 필요한 함수 -----------------

def compute_side_gap_and_overlap_ratio(itemA, itemB, direction, bin):
    """
    direction in ['left','right','front','back'].
    
    - 먼저, itemA.update_adjacent(itemB)를 호출하여, 
    - 그런 다음, itemA.get_nearest_adjacent(direction)를 호출하여,
      만약 인접 아이템이 있다면 그 아이템을 interfering_item으로 사용한다.
    - 그렇지 않으면 itemB를 interfering_item으로 사용한다.
    
    이후, 기존 로직에 따라 gap과 overlap_ratio를 계산한다.
    
    distance < 0 인 경우, 해당 방향에 아이템이 없음을 bin 내벽 거리로 대체하고, overlap_ratio는 0으로 처리한다.
    """
    # 우선 itemA의 인접 목록 업데이트: itemB를 시도
    itemA.update_adjacent(other_item=itemB)
    itemB.update_adjacent(other_item=itemA)

    # Candidate itemA의 데이터
    Ax, Ay, Az = itemA.b_position
    Aw, Ah, Ad = itemA.getDimension()
    A_left   = Ax
    A_right  = Ax + Aw
    A_front  = Ay       # front: 작은 y
    A_back   = Ay + Ah  # back: 큰 y

    # 만약 인접 항목이 있다면, interfering_item으로 사용
    interfering_item = itemA.get_nearest_adjacent(direction)
    if interfering_item is None:
        if direction == 'right':
            distance = bin.width - A_right
        elif direction == 'left':
            distance = A_left
        elif direction == 'front':
            distance = A_front
        elif direction == 'back':
            distance = bin.height - A_back
        overlap_ratio = 0
        return distance, overlap_ratio

    # interfering_item의 데이터
    Bx, By, Bz = interfering_item.b_position
    Bw, Bh, Bd = interfering_item.getDimension()
    B_left   = Bx
    B_right  = Bx + Bw
    B_front  = By
    B_back   = By + Bh

    distance = 0
    overlap_ratio = 0

    if direction == 'front':
        # A의 front = A.y (작은 y)
        # interfering_item의 back = B_back
        distance = A_front - B_back
        # Overlap: x축 (candidate의 x범위: [A_left, A_right])
        overlap = max(0, min(A_right, B_right) - max(A_left, B_left))
        if Aw != 0:
            overlap_ratio = overlap / Aw
    elif direction == 'back':
        # A의 back = A_back
        # interfering_item의 front = B_front
        distance = B_front - A_back
        # Overlap: x축 overlap
        overlap = max(0, min(A_right, B_right) - max(A_left, B_left))
        if Aw != 0:
            overlap_ratio = overlap / Aw
    elif direction == 'left':
        # A의 left = A_left
        # interfering_item의 right = B_right
        distance = A_left - B_right
        # Overlap: y축 overlap (candidate의 y범위: [Ay, A_back])
        overlap = max(0, min(A_back, B_back) - max(Ay, B_front))
        if Ah != 0:
            overlap_ratio = overlap / Ah
    elif direction == 'right':
        # A의 right = A_right
        # interfering_item's left = B_left
        distance = B_left - A_right
        # Overlap: y축 overlap
        overlap = max(0, min(A_back, B_back) - max(Ay, B_front))
        if Ah != 0:
            overlap_ratio = overlap / Ah
    else:
        raise ValueError("direction must be in ['left','right','front','back']")

    # distance < 0 처리: 해당 방향에 아이템이 없다고 간주하고, bin내벽과 candidate의 해당 면 사이의 거리로 대체
    if distance < 0:
        if direction == 'right':
            distance = bin.width - A_right
        elif direction == 'left':
            distance = A_left
        elif direction == 'front':
            distance = A_front
        elif direction == 'back':
            distance = bin.height - A_back
        overlap_ratio = 0

    return distance, overlap_ratio


def shift_bin_coor2world(bin, item_pick_coor, current_rotation):
    """
    bin 내부 좌표계(item_pick_coor) ➜ 월드 좌표계 변환.
    ── 변환 과정 ───────────────────────────────
    1) bin.rotation_quat 으로 item_pick_coor 를 회전
    2) bin.w_position 만큼 병행이동
    3) 집게(또는 TCP)의 “로컬” 후보 자세 중 현재 자세에 가장 가까운 것을 고르고
       bin.rotation_quat 과 곱해 월드‑기준 최종 자세로 변환
    """
    # 1) bin 회전 적용 ----------------
    ix, iy, iz = item_pick_coor[:3]
    # 단위를 mm->m로 변환
    local_vec   = np.array([ix, iy, iz], dtype=float)

    R_bin = quaternion_to_rotation_matrix(np.asarray(bin.rotation_quat, dtype=float))
    # print("Rotation matrix R_bin:\n", R_bin)
    rotated_vec = R_bin @ local_vec          # bin방향 회전

    # 2) bin 평행이동 ----------------
    bx, by, bz  = bin.w_position[:3]
    wx, wy, wz  = rotated_vec + np.array([bx, by, bz], dtype=float) # bin으로 평행이동

    # 3) 점에 대한 bin 평행+ 회전 이동 + 놓는 방향 자세는 item자세로  ----------------
    return [wx, wy, wz,*current_rotation]


# -----------  단위 변환 ----------------

def mm2m(position: list) -> list:
    '''
    mm단위를 m단위로 변환
    '''
    return [position[0]/1000, position[1]/1000, position[2]/1000]


def cm2m(position: list) -> list:
    '''
    cm단위를 m단위로 변환
    '''
    return [position[0]/100, position[1]/100, position[2]/100]


def quaternion_to_rotation_matrix(quat):
    """
    Quaternion [x, y, z, w] → 3x3 Rotation Matrix
    """
    x, y, z, w = quat
    return np.array([
        [1-2*(y**2 + z**2), 2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),     1-2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),   1-2*(x**2 + y**2)]
    ])

def rotate_about_center(vertices: np.ndarray, quat: List[float]) -> np.ndarray:
    """
    ─ vertices : (N,3) array      ─ 로컬 꼭짓점 좌표
    ─ quat     : [x,y,z, w] list   ─ 회전 quaternion

    물체의 **기하학적 중심**(centroid)을 원점으로 옮겨서 회전한 뒤
    다시 되돌려 준다.
    """
    # ① centroid 구하기
    center = vertices.mean(axis=0)           # (3,)

    # ② 원점으로 이동
    v_shift = vertices - center

    # ③ 회전 행렬
    R = quaternion_to_rotation_matrix(quat)  # (3,3)

    # ④ 회전 + 다시 평행이동
    v_rot = (R @ v_shift.T).T + center
    return v_rot


