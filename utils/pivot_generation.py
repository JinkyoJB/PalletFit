# utils/pivot_generation.py
from utils.Pivot import Pivot
from planning.item import RotationType
from planning.itemManager import global_item_manager

from utils.projection import boundary_projection
from utils.get_value import choose_best_rotation
from utils.checkPivot import checkPivot_R
from utils.constants import EPS

import numpy as np
import copy
from copy import deepcopy
import math
from collections import defaultdict

def getFrontLeftSideList(bin, item):
    '''
    Bin의 앞쪽(Front)과 왼쪽(Left) 외곽에 아이템 배치를 시도하여 배치 가능한 후보 리스트를 반환.
    만약 초기 시도에서 후보가 없으면, front_item과 left_item의 ['bottom']에서도 탐색.
    '''

    def check_try_pos_best_rt(bin, item, try_pos):
        best_rt = choose_best_rotation(item, front_item, placement_mode='horizontal')
        alt_rt = RotationType.get_rotation_pair(best_rt)
        for rt in [best_rt, alt_rt]:
            if rt is None:
                continue
            fitted, loaded_item = checkPivot_R(bin, item, try_pos, rt)
            if fitted > 0:
                return loaded_item
        return None
    
    FrontSide_candidate = None
    LeftSide_candidate = None

    all_items = bin.get_all_items()

    # --- Front Side 배치 탐색 ---
    # 가장 앞쪽(y 최소), 가장 오른쪽에 위치한(ex 최대) pivot 선정
    all_items.sort(key=lambda it: (it.b_position[1], -it.ex, it.b_position[2]))
    front_item = all_items[0] if all_items else None

    if front_item:
        try_pos = [front_item.ex + bin.margin_x, front_item.b_position[1], front_item.b_position[2]]
        FrontSide_candidate = check_try_pos_best_rt(bin, item, try_pos)

    # --- Left Side 배치 탐색 ---
    all_items.sort(key=lambda it: (it.b_position[0], -it.ey, it.b_position[2]))
    left_item = all_items[0] if all_items else None

    if left_item:
        try_pos = [left_item.b_position[0], left_item.ey + bin.margin_y, left_item.b_position[2]]
        LeftSide_candidate = check_try_pos_best_rt(bin, item, try_pos)

    return [FrontSide_candidate, LeftSide_candidate]



def project_vertices_left_to_pivots(bin):
    """
    Vertex → ‘왼쪽에서 보이는’ 가장 가까운 right-face 로만 사영해 Pivot 리스트를 만든다.
    반환: 중복 제거된 Pivot 리스트
    """
    mx, my = bin.margin_x, bin.margin_y
    pivots = []

    # ── (A) vertex 제공 목록 ────────────────────────
    vertex_sources = []
    for it in bin.get_visible_items_topdown():
        vertex_sources.append((it.getVertices(), (1, 2, 3, 4, 5, 6,7)))   # 아이템 꼭짓점
    vertex_sources.append((bin.getVertices(), (1, 2)))            # Bin 꼭짓점

    # ── (B) plane 제공 목록 ─────────────────────────
    plane_items = [*bin.get_all_items(), bin]

    # ── (C) 유틸: 한 버텍스에서 ‘가장 가까운’ right-face 찾기 ──
    def _nearest_right_face(vx, vy, vz):
        best = None           # (p_item, xmax, bounds)
        best_xmax = float("-inf")
        for p_item in plane_items:
            plane, bds = p_item.getFaceInfo("right")  # right-face 는 x == xmax
            xmax = round(bds["x"][0], 3)
            # ① 버텍스보다 왼쪽/같은 위치여야 하고
            # ② y, z 가 면 bounds 안에 들어가야 함
            if (vx >= xmax - EPS and
                bds['y'][0]-EPS <= vy <= bds['y'][1]+EPS and
                bds['z'][0]-EPS <= vz <= bds['z'][1]+EPS):
                # 더 ‘오른쪽’(=vx 와 더 가까운) 면이면 갱신
                if xmax > best_xmax:
                    best = (p_item, xmax, bds)
                    best_xmax = xmax
        return best  # 없으면 None

    # ── (D) 각 버텍스별 pivot 생성 ───────────────────
    for verts, vid_list in vertex_sources:
        for vid in vid_list:

            vx, vy, vz = verts[vid]

            nearest = _nearest_right_face(vx, vy, vz)
            if nearest is None:
                continue
            p_item, xmax, bounds = nearest

            # ── 사영점 좌표 ────────────────────────
            base_px = xmax
            base_py, base_pz = vy, vz

            for add_margin in (0, my):                  # y-margin 없이 / y-margin 적용 두 번
                px = base_px
                py = base_py + add_margin
                pz = base_pz

                # ── margin_x 적용 ─────────────────
                if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                    px += mx
                if px >= bin.width - EPS:               # Bin 오른쪽 벽에 붙는 pivot은 제외
                    continue

                # ── 오른쪽에 끼어 있는 아이템과 충돌? ──
                break_flag = False
                for right_id in bin.graph[p_item._id]['right']:
                    right = global_item_manager.get(right_id)
                    if (right.b_position[0] <= px <= right.ex and
                        right.b_position[1] <= py <= right.ey and
                        right.b_position[2] <= pz <  right.ez):
                        break_flag = True
                        break
                if break_flag:
                    continue

                # ── Bin 내부 + 빈자리 + 회전별 pivot 추가 ──
                if (0 <= px <= bin.width and
                    0 <= py <= bin.height and
                    0 <= pz <= bin.depth):
                    if list(bin.search_xyz(px, px + EPS, py, py+EPS, pz+EPS, pz+2*EPS)):
                        continue
                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        pivots.append(
                            Pivot(round(px,3), round(py,3), round(pz,3),
                                  rt,
                                  direction=f'left-{vid}',
                                  bench_bin=bin)
                        )

    # ── (E) 중복 제거 ─────────────────────────────
    uniq_keys, uniq_pivots = set(), []
    for pv in pivots:
        key = (float(pv.x), float(pv.y), float(pv.z), tuple(map(float, pv.rt)))
        if key not in uniq_keys:
            uniq_keys.add(key)
            uniq_pivots.append(pv)

    return uniq_pivots


def project_lines_left_to_pivots(bin, visible_items=None, plane_items=None):
    """
    - 왼쪽 시점에서 보이는 '좌측 모서리 2개'만을 고려.
    - edge 선분(v0, v1)을 Bin/아이템의 right-face들에 직각 투사.
    - 투사된 right-face와 선분 사이( x ∈ (x_face, x_edge] ) 에
      다른 Item AABB 가 하나라도 있으면 **후보에서 제외**.
    - 조건을 모두 통과한 후보 face 에 대해서 투사 결과 점(들)을
      Pivot 으로 변환 후, margin·충돌·중복 검사 과정을 거쳐 리스트 반환.

    Phase 1.1 (2026-05-04): visible_items / plane_items를 외부에서 주입 가능.
                            None이면 기존 동작(직접 호출)으로 fallback.
    """
    mx, my = bin.margin_x, bin.margin_y
    pivots = []
    seen   = set()   # Phase 1.2: 같은 (x, y, z, rt) Pivot 중복 생성 방지

    # ── (A) Edge(line) 소스: 좌측 모서리 2개만 ──────────────────
    EDGE_PAIRS = [
        (1, 5),      # right-back  vertical
        (2, 6),      # right-front   vertical
    ]

    if visible_items is None:
        visible_items = bin.get_visible_items_topdown()

    line_sources = []
    for it in visible_items:
        verts = it.getVertices()
        for i0, i1 in EDGE_PAIRS:
            line_sources.append(
                (it._id, np.asarray(verts[i0], float), np.asarray(verts[i1], float))
            )

    # ── (B) 투사 대상 right-face 들 ────────────────────────────
    if plane_items is None:
        plane_items = [*bin.get_all_items(), bin]

    # ── (C) find right-faces that are NOT blocked ─────────────
    def _candidate_right_faces_line(line_id, v0, v1):
        """
        1) boundary_projection() must return a point/segment (≠None)
        2) In the prism between the projected segment and the edge
           (x_face, x_edge], any foreign AABB → reject.
           * Only the proj segment’s y·z range is used (edge coords excluded)
        """
        cand   = []
        x_edge = max(v0[0], v1[0])          # v0,v1 x값 동일. vertical edge

        for p_item in plane_items:
            # Bin → left face, items → right face
            plane, bds = p_item.getFaceInfo("left" if p_item is bin else "right")
            x_face = round(bds["x"][0], 3)  # 고정 축값
            # if getattr(p_item, '_id', None)== 9 and line_id == 9:
            #     print('debug')
            # face MUST be to the LEFT of the edge (closer to viewer)
            if x_face > x_edge + EPS:        # ❶ face 가 edge 오른쪽이면 skip
                continue

            proj = boundary_projection(plane, bds, v0, v1)
            if proj is None:
                continue

            # unify proj → list of 1 or 2 pts
            proj_pts = [proj] if isinstance(proj[0], (float, np.floating)) else proj

            # ① build prism using ONLY proj segment range (y,z)
            ys = [p[1] for p in proj_pts]
            zs = [p[2] for p in proj_pts]
            y_min, y_max = min(ys), max(ys) 
            z_min, z_max = min(zs), max(zs)
            x_min, x_max = x_edge - EPS, x_face      # (x_edge, x_face)

            # validate thickness before querying r-trees
            if not (x_min < x_max and y_min < y_max and z_min < z_max):
                cand.append((p_item, proj))   # nothing can block
                continue
            
            # ② check blocking items inside the prism
            blocked = False
            for it_id in bin.search_xyz(x_min, x_max, y_min, y_max, z_min, z_max):
                if it_id == line_id:                    # same edge owner
                    continue
                obj = global_item_manager.get(it_id)
                if obj is p_item or not hasattr(obj, "_id"):
                    continue                            # face itself or non-item
                blocked = True
                break

            if not blocked:
                cand.append((p_item, proj))

        return cand
    # ── (D) 후보 face → Pivot 변환 ─────────────────────────────
    for line_id, v0, v1 in line_sources:
        lines_cand_pivots = _candidate_right_faces_line(line_id, v0, v1)

        for p_item, proj in lines_cand_pivots:
            pts = ([proj] if isinstance(proj[0], (float, np.floating)) else [proj[0], proj[1]])

            for (px0, py0, pz0) in pts:
                base_px, base_py, base_pz = float(px0), float(py0), float(pz0)

                def _make_candidate(apply_y_margin: bool):
                    """
                    y-margin(=my)을 우선 적용해보고, 불가하면 None.
                    x-margin(=mx)은 좌/우 벽이 아니면 적용(기존 규칙 유지).
                    margin이 하나라도 적용되면 '위쪽 충돌' 검사 수행.
                    """
                    px, py, pz = base_px, base_py, base_pz
                    y_applied = False

                    # ── y-margin 시도 ──
                    if apply_y_margin:
                        if base_py + my >= bin.height - EPS:   # 상단 넘치면 불가
                            return None
                        py = base_py + my
                        y_applied = abs(my) > EPS

                    # ── x-margin 적용(좌/우 벽 붙어있지 않을 때만) ──
                    x_applied = False
                    if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                        px = px + mx
                        x_applied = abs(mx) > EPS

                    # 오른쪽 벽(px = bin.width) 붙은 pivot 제외(기존 규칙)
                    if px >= bin.width - EPS:
                        return None

                    # ── 위쪽 충돌 검사 ──
                    skip_above_check = (not x_applied) and (not y_applied)
                    if not skip_above_check:
                        if list(bin.search_xyz(px - EPS, px + EPS,
                                               py,       py + EPS,
                                               pz + EPS, pz + 2 * EPS)):
                            return None

                    # ── 경계 체크 ──
                    if not (0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz <= bin.depth):
                        return None

                    return round(px, 3), round(py, 3), round(pz, 3)

                # 1) y-margin 적용 우선
                cand = _make_candidate(True)
                # 2) 불가능하면 무-margin 대체
                if cand is None:
                    cand = _make_candidate(False)
                if cand is None:
                    continue

                px, py, pz = cand

                for rt in RotationType.BasicRotation[0]:
                    if rt is None:
                        continue
                    # Phase 1.2: dedup BEFORE Pivot 생성 (중복 객체 90%+ 제거)
                    rt_key = tuple(map(float, rt))
                    key = (px, py, pz, rt_key)
                    if key in seen:
                        continue
                    seen.add(key)
                    pivots.append(
                        Pivot(px, py, pz,
                              rt,
                              direction='left-edge',
                              options={'face': getattr(p_item, "_id", "bin"), 'line': line_id},
                              bench_bin=bin)
                    )

    return pivots   # 진입 단계에서 이미 dedup됨


def project_vertices_front_to_pivots(bin):
    """
    Vertex → ‘정면에서 보이는’ 가장 가까운 back-face 로만 사영해 Pivot 리스트를 만든다.
    반환값: 중복 제거된 Pivot 리스트
    """
    mx, my = bin.margin_x, bin.margin_y
    pivots = []

    # ── (A) vertex 제공 목록 ─────────────────────────
    vertex_sources = []
    for it in bin.get_visible_items_topdown():
        vertex_sources.append((it.getVertices(), (0,1,2, 3,4,5, 6, 7)))    # 아이템 꼭짓점
    vertex_sources.append((bin.getVertices(), (0, 1)))             # Bin 앞-하단 두 점

    # ── (B) plane 제공 목록 ──────────────────────────
    plane_items = [*bin.get_all_items(), bin]

    # ── (C) 유틸: 한 버텍스에서 가장 가까운 back-face 찾기 ──
    def _nearest_back_face(vx, vy, vz):
        best = None            # (p_item, ymax, bounds)
        best_ymax = float("-inf")
        for p_item in plane_items:
            plane, bds = p_item.getFaceInfo("back")  # back-face 는 y == ymax
            ymax = round(bds["y"][0], 3)
            if (vy >= ymax - EPS and                      # 버텍스가 면 ‘뒤쪽’에 있고
                bds['x'][0]-EPS <= vx <= bds['x'][1]+EPS and
                bds['z'][0]-EPS <= vz <= bds['z'][1]+EPS):  # x,z 가 bounds 내부
                if ymax > best_ymax:                      # 가장 ‘앞쪽’(=vy 와 가장 가까운)
                    best = (p_item, ymax, bds)
                    best_ymax = ymax
        return best  # 없으면 None

    # ── (D) 각 버텍스별 pivot 생성 ──────────────────
    for verts, vid_list in vertex_sources:
        for vid in vid_list:
            vx, vy, vz = verts[vid]

            nearest = _nearest_back_face(vx, vy, vz)
            if nearest is None:
                continue
            p_item, ymax, bounds = nearest

            # ── 사영점 좌표 ───────────────────────
            base_py = ymax
            base_px, base_pz = vx, vz

            for add_margin in (0, my):                # margin_y 미적용 / 적용 두 번
                px = base_px
                py = base_py + add_margin
                pz = base_pz

                # ── margin_x 적용 ────────────────
                if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                    px += mx
                if px >= bin.width - EPS:              # 오른쪽 벽에 붙으면 제외
                    continue

                # ── 뒤쪽(=+Y) 아이템과 충돌? ─────
                skip = False
                for back_id in bin.graph[p_item._id]['back']:
                    back = global_item_manager.get(back_id)
                    if (back.b_position[0] <= px <= back.ex and
                        back.b_position[1] <= py <= back.ey and
                        back.b_position[2] <= pz <  back.ez):
                        skip = True
                        break
                if skip:
                    continue

                # ── Bin 내부 + 빈자리 + 회전별 pivot ──
                if (0 <= px <= bin.width and
                    0 <= py <= bin.height and
                    0 <= pz <= bin.depth):
                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        pivots.append(
                            Pivot(round(px,3), round(py,3), round(pz,3),
                                  rt,
                                  direction=f'front-{vid}',
                                  options={'face': p_item._id},
                                  bench_bin=bin)
                        )

    # ── (E) 중복 제거 ──────────────────────────────
    uniq_keys, uniq_pivots = set(), []
    for pv in pivots:
        key = (float(pv.x), float(pv.y), float(pv.z), tuple(map(float, pv.rt)))
        if key not in uniq_keys:
            uniq_keys.add(key)
            uniq_pivots.append(pv)

    return uniq_pivots

def project_lines_front_to_pivots(bin, visible_items=None, plane_items=None):
    """
    - 앞쪽 시점에서 보이는 '뒤쪽(Back) 모서리 2개'만을 고려.
    - edge 선분(v0, v1)을 Bin/아이템의 back-face들에 직각 투사.
    - 투사된 back-face와 선분 사이( y ∈ (y_face, y_edge] ) 에
      다른 Item AABB 가 하나라도 있으면 **후보에서 제외**.
    - 조건을 모두 통과한 후보 face 에 대해 투사 결과 점(들)을
      Pivot 으로 변환 → margin·충돌·중복 검사 후 리스트 반환.

    Phase 1.1 (2026-05-04): visible_items / plane_items를 외부에서 주입 가능.
    """
    mx, my = bin.margin_x, bin.margin_y
    pivots = []
    seen   = set()   # Phase 1.2

    # ── (A) Edge(line) 소스: 뒤쪽 모서리 2개 ──────────────────────
    EDGE_PAIRS = [
        (3, 7),      # back-left  vertical
        (2, 6),      # back-right vertical
    ]

    if visible_items is None:
        visible_items = bin.get_visible_items_topdown()

    line_sources = []
    for it in visible_items:
        verts = it.getVertices()
        for i0, i1 in EDGE_PAIRS:
            line_sources.append(
                (it._id, np.asarray(verts[i0], float), np.asarray(verts[i1], float))
            )

    # ── (B) 투사 대상 back-face 들 ────────────────────────────────
    if plane_items is None:
        plane_items = [*bin.get_all_items(), bin]

    # ── (C) 선분 하나당 ‘차단 없는’ 후보 back-faces 찾기 ─────────
    def _candidate_back_faces_line(line_id, v0, v1):
        """
        1) boundary_projection() 결과가 None 이 아니어야 함.
        2) ‘모서리 ↔ proj 선분’ 사이 (y_face, y_edge] 구간에
        다른 아이템 AABB 가 하나라도 있으면 후보에서 제외.
        ── x·z 범위는 proj 선분 두 점만으로 한정(모서리 좌표 포함 X)
        """
        cand = []
        y_edge = min(v0[1], v1[1])                 # 모서리 쪽 y

        for p_item in plane_items:
            plane, bds = p_item.getFaceInfo("front" if p_item is bin else "back")
            y_face = round(bds["y"][0], 3)
            # if getattr(p_item, '_id', None)== 9 and line_id == 12:
            #     print('debug')
            # 모서리가 face보다 앞쪽(시야 쪽)에 없으면 skip
            if y_edge < y_face - EPS:
                continue

            proj = boundary_projection(plane, bds, v0, v1)
            if proj is None:
                continue

            # proj_pts 를 리스트(1 or 2 점)로 통일
            proj_pts = [proj] if isinstance(proj[0], (float, np.floating)) else proj

            # ── ① proj 선분의 x·z 범위만 사용해 기둥 영역 설정 ─────────
            xs = [p[0] for p in proj_pts]
            zs = [p[2] for p in proj_pts]
            x_min, x_max = min(xs), max(xs)
            z_min, z_max = min(zs), max(zs)
            y_min, y_max = y_face + EPS, y_edge - EPS

            if not (x_min < x_max and y_min < y_max and z_min < z_max): # 시야 차단 구간 자체가 없음
                cand.append((p_item, proj))
                continue

            # ── ② 기둥 영역 안에 끼어드는 박스가 있는지 검사 ──────────
            blocked = False
            for it_id in bin.search_xyz(x_min, x_max, y_min, y_max, z_min, z_max):
                if it_id == line_id:
                    continue            # 같은 모서리 아이템
                obj = global_item_manager.get(it_id)
                if obj is p_item or not hasattr(obj, "_id"):
                    continue            # face 아이템 본인, _id 없는 객체 무시
                blocked = True
                break

            if not blocked:
                cand.append((p_item, proj))

        return cand

    # ── (D) 후보 face → Pivot 변환 ──────────────────────────────
    for line_id, v0, v1 in line_sources:
        _candidate_points = _candidate_back_faces_line(line_id, v0, v1)

        for p_item, proj in _candidate_points:
            pts = ([proj] if isinstance(proj[0], (float, np.floating)) else [proj[0], proj[1]])

            for (px0, py0, pz0) in pts:
                base_px, base_py, base_pz = float(px0), float(py0), float(pz0)

                def _make_candidate(apply_y_margin: bool):
                    """
                    y-margin(=my)을 우선 적용해보고, 불가하면 None 반환.
                    x-margin(=mx)은 좌/우 벽이 아니면 항상 적용(기존 규칙 유지).
                    margin이 하나라도 적용되면 '위쪽 충돌' 검사 수행.
                    """
                    px, py, pz = base_px, base_py, base_pz
                    y_applied = False

                    # ── y-margin 시도 ──
                    if apply_y_margin:
                        # 뒤쪽 벽(y=height) 넘지 않는지
                        if base_py + my >= bin.height - EPS:
                            return None
                        py = base_py + my
                        y_applied = abs(my) > EPS

                    # ── x-margin 적용(좌/우 벽 붙어있지 않을 때만) ──
                    x_applied = False
                    if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                        px = px + mx
                        x_applied = abs(mx) > EPS

                    # ── 위쪽 충돌 검사 ──
                    skip_above_check = (not x_applied) and (not y_applied)
                    if not skip_above_check:
                        if list(bin.search_xyz(px - EPS, px + EPS,
                                               py,       py + EPS,
                                               pz + EPS, pz + 2 * EPS)):
                            return None

                    # ── 경계 체크 ──
                    if not (0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz <= bin.depth):
                        return None

                    # 뒤쪽 벽(y = height)에 붙은 pivot 제외(기존 규칙 유지)
                    if py >= bin.height - EPS:
                        return None

                    return round(px, 3), round(py, 3), round(pz, 3), x_applied, y_applied

                # 1) y-margin 적용 시도
                cand = _make_candidate(True)
                # 2) 불가능하면 무-margin 대체
                if cand is None:
                    cand = _make_candidate(False)
                if cand is None:
                    continue

                px, py, pz, _, _ = cand

                for rt in RotationType.BasicRotation[0]:
                    if rt is None:
                        continue
                    # Phase 1.2: dedup BEFORE Pivot 생성
                    rt_key = tuple(map(float, rt))
                    key = (px, py, pz, rt_key)
                    if key in seen:
                        continue
                    seen.add(key)
                    pivots.append(
                        Pivot(px, py, pz,
                              rt,
                              direction='front-edge',   # 시선(front) 기준
                              options={'face': getattr(p_item, "_id", "bin"), 'line': line_id},
                              bench_bin=bin)
                    )

    return pivots   # Phase 1.2: 진입 단계에서 이미 dedup됨

def project_vertices_right_to_pivots(bin):
    """
    Vertex → ‘오른쪽에서 보이는’ 가장 가까운 left-face 로만 사영해 Pivot 리스트를 만든다.
    반환값: 중복 제거된 Pivot 리스트
    """
    mx, my, EPS = bin.margin_x, bin.margin_y, 1e-1
    pivots = []

    # ── (A) vertex 목록 ─────────────────────────────
    vertex_sources = []
    for it in bin.get_visible_items_topdown():
        vertex_sources.append((it.getVertices(), (0, 3, 4, 7)))   # (추정) 아이템 왼쪽 면의 꼭짓점
    vertex_sources.append((bin.getVertices(), (0, 3)))            # Bin 왼쪽-하단 두 점

    # ── (B) plane 목록 : 모든 item + Bin ─────────────
    plane_items = [*bin.get_all_items(), bin]

    # ── (C) 한 버텍스에서 가장 가까운 left-face 찾기 ──
    def _nearest_left_face(vx, vy, vz):
        best = None                 # (p_item, xmin, bounds)
        best_xmin = float("-inf")   # “vx 쪽에서 가장 오른쪽” xmin
        for p_item in plane_items:
            plane, bds = p_item.getFaceInfo("left")   # left-face: x == xmin
            xmin = round(bds["x"][0], 3)
            # 버텍스가 face 보다 오른쪽(=vx ≥ xmin)이고, y·z 가 bounds 안이면 후보
            if (vx >= xmin - EPS and
                bds['y'][0]-EPS <= vy <= bds['y'][1]+EPS and
                bds['z'][0]-EPS <= vz <= bds['z'][1]+EPS):
                if xmin > best_xmin:                  # 가장 큰 xmin → 가장 가까운 left-face
                    best = (p_item, xmin, bds)
                    best_xmin = xmin
        return best

    # ── (D) pivot 생성 ─────────────────────────────
    for verts, vid_list in vertex_sources:
        for vid in vid_list:
            vx, vy, vz = verts[vid]

            nearest = _nearest_left_face(vx, vy, vz)
            if nearest is None:
                continue
            p_item, xmin, bounds = nearest

            # --- 사영점 ---
            base_px = xmin
            base_py, base_pz = vy, vz

            for add_margin in (0, my):               # margin_y 미적용 / 적용
                px = base_px
                py = base_py + add_margin
                pz = base_pz

                # ── margin_x 적용 (내부 면이면 왼쪽으로) ──
                if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                    px -= mx                         # 왼쪽(-X)으로 margin 만큼 이동
                if px <= EPS:                        # Bin 왼쪽 벽에 거의 붙으면 skip
                    continue

                # ── 왼쪽(−X) 아이템 충돌 체크 ────────
                skip = False
                for left_id in bin.graph[p_item._id]['left']:
                    left_item = global_item_manager.get(left_id)
                    if (left_item.b_position[0] <= px <= left_item.ex and
                        left_item.b_position[1] <= py <= left_item.ey and
                        left_item.b_position[2] <= pz <  left_item.ez):
                        skip = True
                        break
                if skip:
                    continue

                # ── Bin 내부 + 빈자리 + 회전별 Pivot ──
                if (0 <= px <= bin.width and
                    0 <= py <= bin.height and
                    0 <= pz <= bin.depth):

                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        pivots.append(
                            Pivot(round(px,3), round(py,3), round(pz,3),
                                  rt,
                                  direction=f'right-{vid}',
                                  bench_bin=bin)
                        )

    # ── (E) 중복 제거 ───────────────────────────────
    uniq_keys, uniq_pivots = set(), []
    for pv in pivots:
        key = (float(pv.x), float(pv.y), float(pv.z), tuple(map(float, pv.rt)))
        if key not in uniq_keys:
            uniq_keys.add(key)
            uniq_pivots.append(pv)

    return uniq_pivots


def project_vertices_down_to_pivots(bin):
    """
    Pivot class로 반환
    """
    mx, my = bin.margin_x, bin.margin_y
    pivots = []

    # (A) 상단 꼭짓점
    vsources = []
    for it in bin.get_visible_items_topdown():
        vsources.append((it.getVertices(), (0,1,2,3)))
    vsources.append((bin.getVertices(), (4, 5, 6, 7)))

    # (B) plane : top face (z = zmax)
    planes = [i for i in bin.get_all_items()] + [bin]

    for verts, idxs in vsources:
        for vid in idxs:
            vx, vy, vz = verts[vid]
            for p in planes:
                face_name = 'bottom' if (p is bin) else 'top'
                _, bd  = p.getFaceInfo(face_name)      # ← 여기만 변경
                zmax = bd['z'][0]

                pz, px, py = zmax, vx, vy
                if not(vz > zmax +EPS):
                    continue

                if not (bd['x'][0]-EPS <= px <= bd['x'][1]+EPS and bd['y'][0]-EPS <= py <= bd['y'][1]+EPS):
                    continue

                # vertex와 사영면 사이에 아이템이 있으면 continue ─────── ①
                block_ids = bin.search_xyz(
                    px, px + mx - EPS,               # 단면 x
                    py, py + my - EPS,               # 단면 y
                    zmax + EPS, vz - EPS             # z 범위 (plane 바로 위 ~ vertex 바로 아래)
                )
                # plane 을 이룬 주체(p) 자신은 무시
                block_ids = [bid for bid in block_ids
                             if bid != getattr(p, '_id', None)]
                if block_ids:                        # 뭔가라도 있으면 막혀-있음
                    continue
                
                if pz >= bin.depth:
                    continue

                if 0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz < bin.depth:
                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        pz = round(pz, 3)
                        px = round(px, 3)
                        py = round(py, 3)

                        overlap_ids = list(bin.search_xyz(px, px+ mx - EPS, py, py + my - EPS, pz, pz + EPS))
                        # overlap_id가 존재하고 p._id와 같다면 pivot 생성
                        if np.isclose(pz, 0.0) or (overlap_ids and len(overlap_ids) == 1 and overlap_ids[0] == getattr(p, '_id', None)):
                            # pivot 생성
                            new_pivot = Pivot(px, py, pz, rt, direction=f'down-{vid}', bench_bin=bin)
                            pivots.append(new_pivot)
                        # new_pivot = Pivot(px, py, pz, rt, direction=f'down-{vid}', bench_bin=bin)
                        # pivots.append(new_pivot)

                if not (abs(px - 0) < EPS or abs(px - bin.height) < EPS):
                    px += mx
                if not (abs(py - 0) < EPS or abs(py - bin.width) < EPS):
                    py += my

                # Bin 내부
                if 0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz <= bin.depth:
                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        pz = round(pz, 3)
                        px = round(px, 3)
                        py = round(py, 3)

                        overlap_ids = list(bin.search_xyz(px, px+ mx - EPS, py, py + my - EPS, pz, pz + EPS))
                        # overlap_id가 존재하고 p._id와 같다면 pivot 생성
                        if np.isclose(pz, 0.0) or (overlap_ids and len(overlap_ids) == 1 and overlap_ids[0] == getattr(p, '_id', None)):
                            # pivot 생성
                            new_pivot = Pivot(px, py, pz, rt, direction=f'down-{vid}', bench_bin=bin)
                            pivots.append(new_pivot)
                        # new_pivot = Pivot(px, py, pz, rt, direction=f'down-{vid}', bench_bin=bin)
                        # pivots.append(new_pivot)


    # ── (D) 중복 제거 ─────────────────────────────
    uniq_keys: set[tuple] = set()      # 이미 본 키 저장
    uniq_pivots: list[Pivot] = []      # 결과 리스트    
    for pv in pivots:                  # pivots 는 앞쪽 단계에서 만든 원본 리스트
        # (주의) numpy.float64 → float 로 바꿔야 set 에 넣을 수 있음
        key = (
            float(pv.x),               # x
            float(pv.y),               # y
            float(pv.z),               # z
            tuple(map(float, pv.rt)),   # 회전(quaternion) 4-원소 → 해시가능 튜플
        )
        if key not in uniq_keys:       # 처음 보는 조합이면 keep
            uniq_keys.add(key)
            uniq_pivots.append(pv)
    # 필요하다면 함수의 반환값을 uniq_pivots 로 교체
    return uniq_pivots


def project_lines_down_to_pivots(bin, visible_items=None, plane_items=None):
    '''
    위쪽 시점에서 보이는 '위쪽'모서리 4개만을 고려.
    - edge 선분(v0, v1)을 Bin/아이템의 top-face들에 직각 투사.
    - 투사된 top-face와 선분 사이( z ∈ (z_face, z_edge] ) 에
      다른 Item AABB 가 하나라도 있으면 **후보에서 제외**.
    - 조건을 모두 통과한 후보 face 에 대해 투사 결과 점(들)을
      Pivot 으로 변환 → margin·충돌·중복 검사 후 리스트 반환.

    Phase 1.1 (2026-05-04): visible_items / plane_items를 외부에서 주입 가능.
    '''
    mx, my = bin.margin_x, bin.margin_y
    pivots = []
    seen   = set()   # Phase 1.2

    # ── (A) Edge(line) 소스: 위쪽 모서리 4개 ─────────────────────
    EDGE_PAIRS = [
        (4,5),      # top-left  horizontal
        (6,7),      # top-right horizontal
        (5,6),      # top-front vertical
        (4,7)      # top-back  vertical
    ]

    if visible_items is None:
        visible_items = bin.get_visible_items_topdown()

    line_sources = []
    for it in visible_items:
        verts = it.getVertices()
        for i0, i1 in EDGE_PAIRS:
            line_sources.append(
                (it._id, np.asarray(verts[i0], float), np.asarray(verts[i1], float))
            )

    # ── (B) 투사 대상 back-face 들 ────────────────────────────────
    if plane_items is None:
        plane_items = [*bin.get_all_items(), bin]

    # ── (C) 선분 하나당 ‘차단 없는’ 후보 top-faces 찾기 ───────────
    def _candidate_top_faces_line(line_id, v0, v1):
        """
        1) boundary_projection() 결과가 None 이 아니어야 함.
        2) ‘모서리 ↔ proj 선분’ 사이 (z_face, z_edge] 구간에
        다른 아이템 AABB 가 하나라도 있으면 후보에서 제외.
        ── x·y 범위는 proj 선분 두 점만으로 한정(모서리 좌표 포함 X)
        """
        cand = []
        z_edge = max(v0[2], v1[2])                 # 모서리 쪽 z

        for p_item in plane_items:
            plane, bds = p_item.getFaceInfo("bottom" if p_item is bin else "top")
            z_face = round(bds["z"][0], 3)
            # 모서리가 face보다 아래쪽(=z_edge ≤ z_face)이면 skip
            if z_edge < z_face + EPS:
                continue

            proj = boundary_projection(plane, bds, v0, v1)
            if proj is None:
                continue

            # proj_pts 를 리스트(1 or 2 점)로 통일
            proj_pts = [proj] if isinstance(proj[0], (float, np.floating)) else proj

            # ── ① proj 선분의 x·y 범위만 사용해 기둥 영역 설정 ─────────
            xs = [p[0] for p in proj_pts]
            ys = [p[1] for p in proj_pts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_min, z_max = z_face + EPS, z_edge - EPS

            if not (x_min < x_max and y_min < y_max and z_min < z_max):
                cand.append((p_item, proj))
                continue
            # ── ② 기둥 영역 안에 끼어드는 박스가 있는지 검사 ──────────
            blocked = False
            for it_id in bin.search_xyz(x_min, x_max, y_min, y_max, z_min, z_max):
                if it_id == line_id:
                    continue            # 같은 모서리 아이템
                obj = global_item_manager.get(it_id)
                if obj is p_item or not hasattr(obj, "_id"):
                    continue            # face 아이템 본인, _id 없는 객체 무시
                blocked = True
                break   
            if not blocked:
                cand.append((p_item, proj)) 
        return cand
    
    # ── (D) 후보 face → Pivot 변환 ──────────────────────────────
    for line_id, v0, v1 in line_sources:
        _candidate_points = _candidate_top_faces_line(line_id, v0, v1)
        for p_item, proj in _candidate_points:
            pts = ([proj] if isinstance(proj[0], (float, np.floating))else [proj[0], proj[1]])
            for (px0, py0, pz0) in pts:
                base_px, base_py, base_pz = float(px0), float(py0), float(pz0)

                for add_margin in (0.0, my):                # margin_y 미적용 / 적용 두 번
                    px = base_px
                    py = base_py + add_margin
                    pz = base_pz

                    # margin_x 적용 (좌/우 벽에 붙어 있지 않을 때만)
                    if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                        px += mx
                    if pz >= bin.height - EPS:        # 위쪽 벽(z = height)에 붙은 pivot 제외
                        continue

                    # 해당 pivot의 살짝 위에 있는 위치에 아이템이 이미 존재하는가?
                    # margin_x == 0 and margin_y == 0 이면 '위쪽 충돌' 검사 skip
                    skip_above_check = (abs(mx) < EPS and abs(my) < EPS)
                    if not skip_above_check:
                        # float 오차를 줄이기 위해 x 범위는 ±EPS 로 살짝 여유
                        if list(bin.search_xyz(px - EPS, px + EPS,
                                            py,       py + EPS,
                                            pz + EPS, pz + 2 * EPS)):
                            continue

                    # Bin 경계 체크
                    if (0 <= px <= bin.width and
                        0 <= py <= bin.height and
                        0 <= pz <= bin.depth):
                        rpx, rpy, rpz = round(px, 3), round(py, 3), round(pz, 3)
                        for rt in RotationType.BasicRotation[0]:
                            if rt is None:
                                continue
                            # Phase 1.2: dedup BEFORE Pivot 생성
                            rt_key = tuple(map(float, rt))
                            key = (rpx, rpy, rpz, rt_key)
                            if key in seen:
                                continue
                            seen.add(key)
                            pivots.append(
                                Pivot(rpx, rpy, rpz,
                                      rt,
                                      direction='down-edge',   # 시선(top) 기준
                                      bench_bin=bin)
                            )

    return pivots   # Phase 1.2: 진입 단계에서 이미 dedup됨

def project_vertices_top_to_pivots(bin):
    """
    윗면(top face) 꼭짓점 4개를 기반으로 pivot 좌표를 생성.
    """
    pivots = []

    # ── 1) 꼭짓점 수집
    # for it in bin.get_all_items():
    top_item_ids = bin.get_visible_items_topdown()
    for it in top_item_ids:
        verts = it.getVertices()  # 8×3
        for vid in (4, 5, 6, 7):
        # for vid in (4, 6):
            x, y, z = map(float, verts[vid])

            if (it.b_position[0] < x < it.ex) and (it.b_position[1] < y < it.ey) and (it.b_position[2] < z < it.ez):
                continue

            # Bin 내부 여부 확인
            if 0 <= x <= bin.width and 0 <= y <= bin.height and 0 <= z < bin.depth:
                for rt in RotationType.BasicRotation[0]:
                    if rt is None:
                        continue
                    pz = round(z, 3)
                    px = round(x, 3)
                    py = round(y, 3)
                    # pivot에 완전 쪼그마한 물건이라도 놓을 수 없으면
                    search_list = list(bin.search_xyz(px, px, py, py, pz, pz + EPS))
                    if len(search_list) > 0:
                        continue
                    new_pivot = Pivot(px, py, pz, rt, direction=f'up-{vid}', bench_bin=bin)
                    pivots.append(new_pivot)

    # # ── 2) 중복 제거
    # uniq_keys: set[tuple] = set()      # 이미 본 키 저장
    # uniq_pivots: list[Pivot] = []      # 결과 리스트
    # for pv in pivots:                  # pivots 는 앞쪽 단계에서 만든 원본 리스트
    #     # (주의) numpy.float64 → float 로 바꿔야 set 에 넣을 수 있음
    #     key = (
    #         float(pv.x),               # x
    #         float(pv.y),               # y
    #         float(pv.z),               # z
    #         tuple(map(float, pv.rt)),   # 회전(quaternion) 4-원소 → 해시가능 튜플
    #     )

    #     if key not in uniq_keys:       # 처음 보는 조합이면 keep
    #         uniq_keys.add(key)
    #         uniq_pivots.append(pv)
    # 필요하다면 함수의 반환값을 uniq_pivots 로 교체
    return pivots

def collect_top_edge_candidate_positions(bin, item):
    """
    collect_top_edge_candidate_positions
    Bin의 상단(Top) 외곽에 아이템 배치를 시도하여 배치 가능한 후보 리스트를 반환.
    """

    candidates = []
    group_idx = RotationType.get_BasicRotation_index(item.rotation_quat)
    rotations = RotationType.BasicRotation[group_idx]

    # 각 pivot에서 배치 가능성 탐색
    # for item_in_bin in bin.get_top_items():
    for item_in_bin in bin.get_visible_items_topdown():
        for rt in rotations:
            rotated_item = copy.deepcopy(item)
            rotated_item.rotation_quat = rt

            # 회전된 아이템 치수
            itw, ith, itd = rotated_item.getDimension()
            # -----------------------------
            # (1) 왼쪽  앞 상단 꼭지점
            # -----------------------------
            pos_1 = [item_in_bin.b_position[0], item_in_bin.b_position[1], item_in_bin.ez]
            candidates.append(pos_1)

            # -----------------------------
            # (2) 오른쪽 앞 상단 꼭지점
            # -----------------------------
            pos_2 = (item_in_bin.ex - itw, item_in_bin.b_position[1], item_in_bin.ez)
            candidates.append(pos_2)
            # -----------------------------
            # (3) 왼쪽 뒤 상단 꼭지점
            # -----------------------------
            pos_3 = (item_in_bin.b_position[0], item_in_bin.ey - ith, item_in_bin.ez)
            candidates.append(pos_3)

            # -----------------------------
            # (4) 오른쪽 뒤 상단 꼭지점
            # -----------------------------
            pos_4 = (item_in_bin.ex - itw, item_in_bin.ey - ith, item_in_bin.ez)
            candidates.append(pos_4)
            # fitted_4, loaded_item_4 = _check_and_record(bin, rotated_item, pos_4, rt)
            # if fitted_4:
            #     candidates.append(loaded_item_4)

    return candidates

def project_vertices_back_to_pivots(bin):
    """
    ① 모든 아이템 0·1번 꼭짓점 + Bin 꼭짓점 0·1
       (y-축 기준 ‘뒤쪽’ 하단 두 점)
    ② 각 아이템(front face)·Bin(front face) 로 -Y(뒤→앞) 사영
    ③ 조건·margin 만족 pivot 반환
    """
    mx, my= bin.margin_x, bin.margin_y
    pivots = []

    vsources = []
    for it in bin.get_all_items():
        vsources.append((it.getVertices(), (2, 3)))          # 뒤쪽 하단 (y=h)
    vsources.append((bin.getVertices(), (0, 1)))

    planes = [i for i in bin.get_all_items()] + [bin]

    for verts, idxs in vsources:
        for vid in idxs:
            vx, vy, vz = verts[vid]
            for p in planes:
                plane, bd = p.getFaceInfo('front')           # y = ymin
                ymin = bd['y'][0]
                if vy <= ymin + EPS:
                    continue
                py = ymin
                px, pz = vx, vz
                if not (bd['x'][0]-EPS <= px <= bd['x'][1]+EPS and
                        bd['z'][0]-EPS <= pz <= bd['z'][1]+EPS):
                    continue
                if not (abs(py) < EPS or abs(py - bin.depth) < EPS):
                    py -= my
                if 0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz <= bin.depth:
                    pivots.append([float(px), float(py), float(pz)])

    uniq, seen = [], set()
    for p in pivots:
        k = tuple(round(v, 4) for v in p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


# ───── CP (Corner‑Point) ──────────────────────────────────────────
def get_pivots_cp(bin_obj):
    """
    Huang et al.(2015) Corner‑Points  +  margin(x,y) 보정 버전
    ----------------------------------------------------------
    * ‘위에서 보이는’ 아이템의 8 vertices 를 후보로 삼음
    * margin_x : x‑축 이웃 간격 (좌 / 우 방향으로 ±)
      ─ (0 또는 bin.width 벽에 붙은 vertex 는 제외)
    * margin_y : y‑축 이웃 간격 (아래 → 그대로,   위쪽 → +margin_y)
    * z‑축은 별도 margin 없이 그대로 사용
    """
    # 빈 박스일 경우: 원점 기준 허용 회전 2개만 반환
    if getattr(bin_obj, "origin_item_id", None) is None:
        piv0 = []
        for rid in [0, 1]:
            rt = RotationType.index_to_quat(rid)
            if rt is not None:
                piv0.append(Pivot(0, 0, 0, rt, bench_bin=bin_obj))
        return piv0
    
    mx, my = bin_obj.margin_x, bin_obj.margin_y
    pivots, uniq = [], set()

    # ① 지금 보이는(top‑down) 아이템만 선택
    for it in bin_obj.get_visible_items_topdown():
        vtx = it.getVertices()                # 8×3

        for vid, (vx,vy,vz) in enumerate(vtx):
            # ── Bin 경계 내부 + 빈공간 체크 ──────────────────────
            if not (0 <= vx <= bin_obj.width
                    and 0 <= vy <= bin_obj.height
                    and 0 <= vz <= bin_obj.depth):
                continue
            if bin_obj.search_xyz(vx+EPS,vx+EPS,vy+EPS,vy+EPS,vz+EPS,vz+2*EPS):
                continue

            # ── margin_y : 0 / +my 두 버전 ────────────────────
            for add_my in (0.0, my):
                py = vy + add_my
                if py > bin_obj.height - EPS:    # 상단 벽 돌파 → skip
                    continue

                # ── margin_x : 좌우 방향으로 밀기 ────────────
                #   * bin 좌벽/우벽에 붙은 vertex 는 건드리지 않음
                if np.isclose(vx, 0.0, atol=EPS) or np.isclose(vx, bin_obj.width, atol=EPS):
                    px_list = [vx]               # 그대로
                else:
                    # 내부 vertex → “왼쪽면” 인지 “오른쪽면” 인지 판단
                    px_list = [vx + mx]      # 오른쪽 블록이면 −mx (← 내부로)
                    # margin 결과가 벽 넘어가면 skip
                    px_list = [p for p in px_list if EPS <= p <= bin_obj.width-EPS]

                # ── PX 들에 대해 회전별 Pivot 생성 ─────────────
                for px in px_list:
                    pz = vz
                    for rt in RotationType.BasicRotation[0]:
                        key = (round(px,4),round(py,4),round(pz,4),
                               tuple(map(float,rt)))
                        if key in uniq:
                            continue
                        uniq.add(key)
                        pivots.append(
                            Pivot(px, py, pz, rt,
                                  direction=f'cp-{vid}',
                                  bench_bin=bin_obj)
                        )

    return pivots

# ──── EP (Extreme‑Point)  (bounds‑safe 버전) ──────────────────────
def get_pivots_ep(bin_obj):

    # 빈 박스일 경우: 원점 기준 허용 회전 2개만 반환
    if getattr(bin_obj, "origin_item_id", None) is None:
        piv0 = []
        for rid in [0, 1]:
            rt = RotationType.index_to_quat(rid)
            if rt is not None:
                piv0.append(Pivot(0, 0, 0, rt, bench_bin=bin_obj))
        return piv0
    
    mx, my = bin_obj.margin_x, bin_obj.margin_y
    eps    = 1e-4
    pivots, key_set = [], set()

    # ──────────────────────────────────────────────────────────
    def _try_add(px, py, pz, tag):
        if not (0 <= px <= bin_obj.width
                and 0 <= py <= bin_obj.height
                and 0 <= pz <= bin_obj.depth):
            return
        if bin_obj.search_xyz(px+eps, px+eps, py+eps, py+eps,
                              pz+eps, pz+2*eps):
            return
        for rt in RotationType.BasicRotation[0]:
            k = (round(px,4), round(py,4), round(pz,4),
                 tuple(map(float, rt)))
            if k in key_set:
                continue
            key_set.add(k)
            pivots.append(
                Pivot(px, py, pz, rt,
                      direction=tag,
                      bench_bin=bin_obj)
            )

    # ──────────────────────────────────────────────────────────
    def _cast_neg_axis(x0, y0, z0, axis):
        """
        축 ‘axis’ 방향으로 음(-)쪽 레이캐스트 후
        가장 가까운 면 좌표(컨테이너 포함)를 돌려준다.
        없으면 0.0 반환.
        """
        if axis == 'x':
            lo, hi = 0.0, x0 - eps
            if hi < lo:                # 역전 → 빈 공간
                return 0.0
            ids = bin_obj.search_xyz(lo, hi, y0, y0, z0, z0+eps)
            bound = 0.0
            for iid in ids:
                it = global_item_manager.get(iid)
                if it is not None:
                    bound = max(bound, it.ex)
            return bound

        if axis == 'y':
            lo, hi = 0.0, y0 - eps
            if hi < lo:
                return 0.0
            ids = bin_obj.search_xyz(x0, x0, lo, hi, z0, z0+eps)
            bound = 0.0
            for iid in ids:
                it = global_item_manager.get(iid)
                if it is not None:
                    bound = max(bound, it.ey)
            return bound

        if axis == 'z':
            lo, hi = 0.0, z0 - eps
            if hi < lo:
                return 0.0
            ids = bin_obj.search_xyz(x0, x0, y0, y0, lo, hi)
            bound = 0.0
            for iid in ids:
                it = global_item_manager.get(iid)
                if it is not None:
                    bound = max(bound, it.ez)
            return bound
        return 0.0

    # ──────────────────────────────────────────────────────────
    for it in bin_obj.get_visible_items_topdown():
        vx, vy, vz = it.b_position
        w, h, d    = it.getDimension()

        # --- vid 1 : (W,0,0)  –y / –z -------------------------
        bx, by, bz = vx + w, vy, vz
        py = _cast_neg_axis(bx + mx, by, bz, 'y')
        _try_add(bx + mx, py, bz, 'ep')
        _try_add(bx + mx, by + my, bz, 'ep')

        pz = _cast_neg_axis(bx, by, bz, 'z')
        _try_add(bx, by, pz, 'ep')

        # --- vid 3 : (0,H,0)  –x / –z -------------------------
        bx, by, bz = vx, vy + h, vz
        px = _cast_neg_axis(bx, by + my, bz, 'x')
        _try_add(px, by + my, bz, 'ep')
        _try_add(px + mx, by + my, bz, 'ep')

        pz = _cast_neg_axis(bx, by, bz, 'z')
        _try_add(bx, by, pz, 'ep')

        # --- vid 4 : (0,0,D)  –x / –y -------------------------
        bx, by, bz = vx, vy, vz + d
        px = _cast_neg_axis(bx, by, bz, 'x')

        py = _cast_neg_axis(bx, by, bz, 'y')
        _try_add(bx, py, bz, 'ep')

    return pivots


# ───── EMS (Empty‑Maximal‑Space) 기반 Pivot 생성 ──────────────────

class EMSBox:
    """남아‑있는 빈 직육면체(좌표축 정렬)"""
    __slots__ = ("x0","y0","z0","x1","y1","z1")
    def __init__(self, x0,y0,z0, x1,y1,z1):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.x1, self.y1, self.z1 = x1, y1, z1
    def size(self):     # 가로, 세로, 높이
        return self.x1-self.x0, self.y1-self.y0, self.z1-self.z0
    def volume(self):
        dx,dy,dz = self.size();   return dx*dy*dz
    def intersects(self, item):
        ix0, iy0, iz0 = item.b_position
        ix1, iy1, iz1 = item.ex, item.ey, item.ez
        return not ( self.x1 <= ix0 or self.x0 >= ix1 or
                     self.y1 <= iy0 or self.y0 >= iy1 or
                     self.z1 <= iz0 or self.z0 >= iz1 )
    def contains_point(self, x,y,z):
        return self.x0 <= x <= self.x1 and \
               self.y0 <= y <= self.y1 and \
               self.z0 <= z <= self.z1
    def __repr__(self):
        return f"EMS({self.x0},{self.y0},{self.z0}→{self.x1},{self.y1},{self.z1})"

def _split_ems_by_item(ems: EMSBox, item):
    """
    단일 EMS 를 아이템과의 교차를 기준으로 최대 6 조각으로 쪼갠다
    (Martello & Vigo 1998 방법).
    교차가 없으면 원 EMS 1개만 그대로 반환.
    """
    if not ems.intersects(item):
        return [ems]

    ix0, iy0, iz0 = item.b_position
    ix1, iy1, iz1 = item.ex, item.ey, item.ez
    pieces = []

    # 좌
    if ems.x0 < ix0 < ems.x1:
        pieces.append(EMSBox(ems.x0, ems.y0, ems.z0, ix0,   ems.y1, ems.z1))
        ems.x0 = ix0
    # 우
    if ems.x0 < ix1 < ems.x1:
        pieces.append(EMSBox(ix1,   ems.y0, ems.z0, ems.x1, ems.y1, ems.z1))
        ems.x1 = ix1
    # 앞
    if ems.y0 < iy0 < ems.y1:
        pieces.append(EMSBox(ems.x0, ems.y0, ems.z0, ems.x1, iy0,   ems.z1))
        ems.y0 = iy0
    # 뒤
    if ems.y0 < iy1 < ems.y1:
        pieces.append(EMSBox(ems.x0, iy1,   ems.z0, ems.x1, ems.y1, ems.z1))
        ems.y1 = iy1
    # 아래
    if ems.z0 < iz0 < ems.z1:
        pieces.append(EMSBox(ems.x0, ems.y0, ems.z0, ems.x1, ems.y1, iz0))
        ems.z0 = iz0
    # 위
    if ems.z0 < iz1 < ems.z1:
        pieces.append(EMSBox(ems.x0, ems.y0, iz1,   ems.x1, ems.y1, ems.z1))
        ems.z1 = iz1

    return [p for p in pieces if p.volume() > 0]   # 0 부피 제거

# ────────────────────────────────────────────────────────────────
def get_pivots_ems(bin_obj):
    """
    • 초기 EMS = Bin 전체 한 박스  
    • 아이템을 하나씩 보며 EMS 를 split & prune  
    • 각 EMS 원점(x0,y0,z0) 을 Pivot 후보로 사용  
    • margin_x, margin_y 적용 :  
        – x0>0 → 내부 EMS 면들은 (x0+mx, …),  
        – y0>0 → (…, y0+my, …),  
      벽(x0==0 등)은 margin 미적용
    """
    # 빈 박스일 경우: 원점 기준 허용 회전 2개만 반환
    if getattr(bin_obj, "origin_item_id", None) is None:
        piv0 = []
        for rid in [0, 1]:
            rt = RotationType.index_to_quat(rid)
            if rt is not None:
                piv0.append(Pivot(0, 0, 0, rt, bench_bin=bin_obj))
        return piv0
    
    mx, my, eps = bin_obj.margin_x, bin_obj.margin_y, 1e-4
    # 1) EMS 리스트 갱신
    ems_list : list[EMSBox] = [ EMSBox(0,0,0,
                                       bin_obj.width,
                                       bin_obj.height,
                                       bin_obj.depth) ]
    for iid in bin_obj.item_ids:
        item = global_item_manager.get(iid)
        if item is None: continue
        new_list = []
        for ems in ems_list:
            new_list.extend( _split_ems_by_item(deepcopy(ems), item) )
        # 작은 EMS 제거(선택) & 포함관계 제거 가능
        ems_list = new_list

    # 2) EMS → Pivot 변환
    key_set, pivots = set(), []
    for ems in ems_list:
        px = ems.x0 if np.isclose(ems.x0,0.0,atol=eps) else ems.x0 + mx
        py = ems.y0 if np.isclose(ems.y0,0.0,atol=eps) else ems.y0 + my
        pz = ems.z0                                  # z‑margin 없음

        # EMS 가 너무 얇아 margin 적용 시 경계 초과할 수도 있음
        if px > bin_obj.width - eps or py > bin_obj.height - eps:
            continue

        # EMS 안에 최소한 “바늘 상자” 들어갈 정도?
        if bin_obj.search_xyz(px+eps, px+eps, py+eps, py+eps, pz+eps, pz+2*eps):
            continue

        for rt in RotationType.BasicRotation[0]:
            k = (round(px,4), round(py,4), round(pz,4),
                 tuple(map(float,rt)))
            if k in key_set:      # 중복 방지
                continue
            key_set.add(k)
            pivots.append(
                Pivot(px, py, pz, rt,
                      direction='ems',
                      bench_bin=bin_obj)
            )
    return pivots

def project_lines_down_to_pivots2left(bin, pivots):
    '''
    '''
    mx, my = bin.margin_x, bin.margin_y
    candidates = []
    seen       = set()   # Phase 1.2

    for pv in pivots:
        pv_x = max(0, pv.x - EPS)
        # pv.x보다 작거나 같은 곳에 있는 아이템을 찾는다
        block_ids = bin.search_xyz(
            0, pv_x,               # 단면 x
            pv.y, pv.y,               # 단면 y
            pv.z, pv.z             # z 범위 (plane 바로 위 ~ vertex 바로 아래)
        )

        # block_ids가 존재한다면, 아이템들 중 제일 오른쪽에있는 아이템의 오른면에 점을 사영한다.
        if block_ids:
            right_most = 0
            for bid in block_ids:
                obj = global_item_manager.get(bid)
                if obj is not None:
                    right_most = max(right_most, obj.ex)
            base_px, base_py, base_pz = right_most, pv.y, pv.z
        
            for add_margin in (0.0, my):
                px = base_px
                py = base_py + add_margin
                pz = base_pz

                # margin_x 적용
                if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                    px += mx
                if px >= bin.width - EPS:        # 오른쪽 벽 붙은 pivot 제외
                    continue

                # 해당 pivot의 살짝 위에 있는 위치에 아이템이 이미 존재하는가?
                # margin_x == 0 and margin_y == 0 이면 '위쪽 충돌' 검사 skip
                skip_above_check = (abs(mx) < EPS and abs(my) < EPS)
                if not skip_above_check:
                    # float 오차를 줄이기 위해 x 범위는 ±EPS 로 살짝 여유
                    if list(bin.search_xyz(px - EPS, px + EPS,
                                        py,       py + EPS,
                                        pz + EPS, pz + 2 * EPS)):
                        continue

                # 빈자리 검사
                if (0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz <= bin.depth):
                    rpx, rpy, rpz = round(px, 3), round(py, 3), round(pz, 3)
                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        # Phase 1.2: dedup BEFORE Pivot 생성
                        rt_key = tuple(map(float, rt))
                        key = (rpx, rpy, rpz, rt_key)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            Pivot(rpx, rpy, rpz,
                                    rt,
                                    direction='left2-edge',
                                    bench_bin=bin)
                        )

        else:   # block_ids가 없다면, bin의 왼쪽 벽에 점을 사영한다.
            base_px, base_py, base_pz = 0, pv.y, pv.z
        
            for add_margin in (0.0, my):
                px = base_px
                py = base_py + add_margin
                pz = base_pz

                # margin_x 적용
                if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                    px += mx
                if px >= bin.width - EPS:        # 오른쪽 벽 붙은 pivot 제외
                    continue
                # 해당 pivot의 살짝 위에 있는 위치에 아이템이 이미 존재하는가?
                # margin_x == 0 and margin_y == 0 이면 '위쪽 충돌' 검사 skip
                skip_above_check = (abs(mx) < EPS and abs(my) < EPS)
                if not skip_above_check:
                    # float 오차를 줄이기 위해 x 범위는 ±EPS 로 살짝 여유
                    if list(bin.search_xyz(px - EPS, px + EPS,
                                        py,       py + EPS,
                                        pz + EPS, pz + 2 * EPS)):
                        continue
                # 빈자리 검사
                if (0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz <= bin.depth):
                    rpx, rpy, rpz = round(px, 3), round(py, 3), round(pz, 3)
                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        # Phase 1.2: dedup BEFORE Pivot 생성
                        rt_key = tuple(map(float, rt))
                        key = (rpx, rpy, rpz, rt_key)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            Pivot(rpx, rpy, rpz,
                                    rt,
                                    direction='left2-edge',
                                    bench_bin=bin)
                        )
    # ── (E) 중복 제거 후 반환 ─────────────────────────────────
    uniq_keys, uniq_pivots = set(), []
    for pv in candidates:
        key = (float(pv.x), float(pv.y), float(pv.z),
               tuple(map(float, pv.rt)))
        if key not in uniq_keys:
            uniq_keys.add(key)
            uniq_pivots.append(pv)

    return uniq_pivots


def project_lines_down_to_pivots2front(bin, pivots):
    '''
    '''
    mx, my = bin.margin_x, bin.margin_y
    candidates = []
    seen       = set()   # Phase 1.2
    for pv in pivots:
        pv_y = max(0, pv.y - EPS)
        # pv.y보다 작거나 같은 곳에 있는 아이템을 찾는다
        block_ids = bin.search_xyz(
            pv.x, pv.x,               # 단면 x
            0, pv_y,               # 단면 y
            pv.z, pv.z             # z 범위 (plane 바로 위 ~ vertex 바로 아래)
        )
        # block_ids가 존재한다면, 아이템들 중 제일 앞에있는 아이템의 앞면에 점을 사영한다.
        if block_ids:
            front_most = 0
            for bid in block_ids:
                obj = global_item_manager.get(bid)
                if obj is not None:
                    front_most = max(front_most, obj.ey)
            base_px, base_py, base_pz = pv.x, front_most, pv.z
        
            for add_margin in (0.0, my):
                px = base_px + add_margin
                py = base_py
                pz = base_pz

                # margin_x 적용
                if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                    px += mx
                if px >= bin.width - EPS:        # 오른쪽 벽 붙은 pivot 제외
                    continue

                # 해당 pivot의 살짝 위에 있는 위치에 아이템이 이미 존재하는가?
                # margin_x == 0 and margin_y == 0 이면 '위쪽 충돌' 검사 skip
                skip_above_check = (abs(mx) < EPS and abs(my) < EPS)
                if not skip_above_check:
                    # float 오차를 줄이기 위해 x 범위는 ±EPS 로 살짝 여유
                    if list(bin.search_xyz(px - EPS, px + EPS,
                                        py,       py + EPS,
                                        pz + EPS, pz + 2 * EPS)):
                        continue

                # 빈자리 검사
                if (0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz <= bin.depth):
                    rpx, rpy, rpz = round(px, 3), round(py, 3), round(pz, 3)
                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        # Phase 1.2: dedup BEFORE Pivot 생성
                        rt_key = tuple(map(float, rt))
                        key = (rpx, rpy, rpz, rt_key)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            Pivot(rpx, rpy, rpz,
                                    rt,
                                    direction='front2-edge',
                                    bench_bin=bin)
                        )

        else:   # block_ids가 없다면, bin의 앞쪽 벽에 점을 사영한다.
            base_px, base_py, base_pz = pv.x, 0, pv.z
        
            for add_margin in (0.0, my):
                px = base_px + add_margin
                py = base_py
                pz = base_pz

                # margin_x 적용
                if not (abs(px) < EPS or abs(px - bin.width) < EPS):
                    px += mx
                if px >= bin.width - EPS:        # 오른쪽 벽 붙은 pivot 제외
                    continue
                # 해당 pivot의 살짝 위에 있는 위치에 아이템이 이미 존재하는가?
                # margin_x == 0 and margin_y == 0 이면 '위쪽 충
                skip_above_check = (abs(mx) < EPS and abs(my) < EPS)
                if not skip_above_check:
                    # float 오차를 줄이기 위해 x 범위는 ±EPS 로 살짝 여유
                    if list(bin.search_xyz(px - EPS, px + EPS,
                                        py,       py + EPS,
                                        pz + EPS, pz + 2 * EPS)):
                        continue
                # 빈자리 검사
                if (0 <= px <= bin.width and 0 <= py <= bin.height and 0 <= pz <= bin.depth):
                    rpx, rpy, rpz = round(px, 3), round(py, 3), round(pz, 3)
                    for rt in RotationType.BasicRotation[0]:
                        if rt is None:
                            continue
                        # Phase 1.2: dedup BEFORE Pivot 생성
                        rt_key = tuple(map(float, rt))
                        key = (rpx, rpy, rpz, rt_key)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            Pivot(rpx, rpy, rpz,
                                    rt,
                                    direction='front2-edge',
                                    bench_bin=bin)
                        )
    # ── (E) 중복 제거 후 반환 ─────────────────────────────────
    uniq_keys, uniq_pivots = set(), []
    for pv in candidates:       # candidates 는 앞쪽 단계에서 만든 원본 리스트
        key = (float(pv.x), float(pv.y), float(pv.z),
               tuple(map(float, pv.rt)))
        if key not in uniq_keys:
            uniq_keys.add(key)
            uniq_pivots.append(pv)
    return uniq_pivots


def merge_close_pivots(pivots, tol_mm: float = 0.0, keep: str = "fisrt"):
    """
    같은 회전(rt)끼리만 비교.
    두 피벗 간 유클리드 거리 ≤ tol_mm 이면 하나로 병합.
    keep='avg'면 좌표 평균, 'first'면 최초 피벗 좌표 유지.
    """
    if not pivots:
        return []

    # 회전별로 나눠서 병합 (서로 다른 rt는 병합 금지)
    by_rot = defaultdict(list)
    for pv in pivots:
        rt_key = tuple(map(float, pv.rt)) if hasattr(pv, "rt") else None
        by_rot[rt_key].append(pv)

    merged_all = []

    for rt_key, group in by_rot.items():
        # 아직 병합되지 않은 pivot 목록
        unvisited = [True] * len(group)
        i = 0
        while i < len(group):
            if not unvisited[i]:
                i += 1
                continue

            # 새 클러스터 시작
            cluster_idx = [i]
            unvisited[i] = False

            xi, yi, zi = float(group[i].x), float(group[i].y), float(group[i].z)

            # 나머지와 거리 비교 (O(n^2) 간단 버전)
            for j in range(i + 1, len(group)):
                if not unvisited[j]:
                    continue
                xj, yj, zj = float(group[j].x), float(group[j].y), float(group[j].z)
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                if math.sqrt(dx*dx + dy*dy + dz*dz) <= tol_mm:
                    cluster_idx.append(j)
                    unvisited[j] = False

            # 대표 pivot 생성
            base = group[cluster_idx[0]]
            if keep == "avg" and len(cluster_idx) > 1:
                sx = sy = sz = 0.0
                for k in cluster_idx:
                    sx += float(group[k].x)
                    sy += float(group[k].y)
                    sz += float(group[k].z)
                mx, my, mz = sx / len(cluster_idx), sy / len(cluster_idx), sz / len(cluster_idx)
                new_pv = base.__class__(
                    round(mx, 3), round(my, 3), round(mz, 3),
                    base.rt,
                    direction=getattr(base, "direction", None),
                    options=getattr(base, "options", None),
                    bench_bin=getattr(base, "bench_bin", None),
                )
            else:
                # 첫 번째 유지 (좌표만 깔끔하게 반올림)
                new_pv = base
                new_pv.x = round(float(new_pv.x), 3)
                new_pv.y = round(float(new_pv.y), 3)
                new_pv.z = round(float(new_pv.z), 3)

            merged_all.append(new_pv)
            i += 1

    # 동일 좌표·회전 완전 중복 제거 한 번 더
    seen, out = set(), []
    for pv in merged_all:
        key = (float(pv.x), float(pv.y), float(pv.z),
               tuple(map(float, pv.rt)) if hasattr(pv, "rt") else None)
        if key in seen:
            continue
        seen.add(key)
        out.append(pv)

    return out