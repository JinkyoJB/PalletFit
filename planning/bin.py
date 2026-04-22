# planning/bin.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np
from bisect import bisect_left, bisect_right
from rtree import index
from itertools import count

from planning.item import Item, RotationType
from planning.itemManager import global_item_manager
from utils.painter import PainterPlot
from utils.overlap import overlap_intervals  # ← 새로 추가
from utils.constants import BIG_NEG, BIG_POS, EPS
from utils.get_value import get_top_face_occupancy

__all__ = ["Bin"]


class Bin:
    """단일 컨테이너 + 아이템 인덱싱/그래프 관리"""
    # ────────────────────────────────────────────────────────────────────
    # 1) 초기화 & 단위 변환
    # ────────────────────────────────────────────────────────────────────
    def __init__(self,
                 *,
                 width: float,
                 height: float,
                 depth: float,
                 unit: str = "mm",
                 partno: str | None = None,
                 name: str = "unknown",
                 max_weight: float | None = None,
                 w_position: List[float] | None = None,
                 rotation_quat: List[float] | None = None,
                 margin_x: float = 0.0,
                 margin_y: float = 0.0,
                 options: Dict | None = None,
                 corner: float = 0.0,
                 support_surface_ratio: float = 0.5) -> None:
        # 필수 치수
        self.width  = width
        self.height = height
        self.depth  = depth
        self.unit   = unit

        # 추가 메타
        self.partno        = partno
        self.name          = name
        self.max_weight    = max_weight
        self.w_position    = w_position
        self.rotation_quat = rotation_quat or RotationType.RT_WHD
        self.margin_x      = margin_x
        self.margin_y      = margin_y
        self.options       = options or {}
        self.corner        = corner
        self.support_surface_ratio = support_surface_ratio
        self.b_position    = [0, 0, 0]
        self.unfit_items: List[int] = []  # 배치 불가능한 아이템 목록

        # 단위 보정
        if self.unit == "m":   self._m2mm()
        elif self.unit == "cm": self._cm2mm()

        # ---------------- 저장소 자료구조 ----------------
        self.item_ids: List[int] = []             # bin 내부 모든 item_id

        self.index_xy = index.Index()             # 2D R-tree x‑y
        self.index_xz = index.Index()             # 2D R-tree x‑z
        self.index_yz = index.Index()             # 2D R-tree y‑z
        self.xy_map: Dict[int, Tuple[float, float, float, float]] = {}
        self.xz_map: Dict[int, Tuple[float, float, float, float]] = {}
        self.yz_map: Dict[int, Tuple[float, float, float, float]] = {}

        # 인접 그래프 (direction ↔ set(item_id))
        self.graph: Dict[int, Dict[str, Set[int]]] = {}
        self.origin_item_id: int | None = None

        # 생성된 pivotTree 저장용
        self.pivotTree=None

        # 면 캐시
        self._face_info: Dict[str, Tuple[Tuple[float, float, float, float], Dict[str, Tuple[float, float]]]] = {}

    # ------------------------------------------------------------------
    # 1) 단위 변환 내부 헬퍼
    # ------------------------------------------------------------------
    def _m2mm (self):
        self.width *= 1000; self.height *= 1000; self.depth *= 1000; self.unit = "mm"

    def _cm2mm(self):
        self.width *= 10;   self.height *= 10;   self.depth *= 10;   self.unit = "mm"

    # ------------------------------------------------------------------
    # 2) 프로퍼티 & 기본 정보
    # ------------------------------------------------------------------
    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @property
    def size(self) -> int:
        """
        bin 안 ‘실제 박스’의 개수를 반환한다.

        • leaf  아이템  → 1 개
        • composite 아이템 → children_ids 개수
        """
        cnt = 0
        for iid in self.item_ids:
            itm = global_item_manager.get(iid)
            if itm is None:
                continue
            if getattr(itm, "is_composite", False):
                cnt += len(itm.children_ids)
            else:
                cnt += 1
        return cnt

    @property
    def leftover(self) -> float:
        return self.volume - self._total_volume()
    
    @property
    def SU(self) -> float:
        """Space Utilization (SU)"""
        try:
            if self.volume == 0:
                return 0.0
            return max(0.0 ,min(self._total_volume() / self.volume, 1.0))
        except Exception:
            return 0.0
    
    # ------------------------------------------------------------------
    # 2-1) 총 중량, 부피 헬퍼 
    # ------------------------------------------------------------------
    def _total_weight(self):
        return sum(global_item_manager.get(iid).weight for iid in self.item_ids
                   if global_item_manager.get(iid))

    def _total_volume(self):
        return sum(global_item_manager.get(iid).volume for iid in self.item_ids
                   if global_item_manager.get(iid))

    # ------------------------------------------------------------------
    # 3) 외부 API
    # ------------------------------------------------------------------
    def store(self, target: Item):
        """
        bin에 아이템을 저장합니다.
        item정보의 기준은 무.조.건. global_item_manager에 등록된 값입니다!
        🟠bin에 저장하기 전 반드시 global_item_manager 정보를 확인하세요!🟠
        """

        # 1. 위치가 고정되었으니, 아이템 면정보 캐싱
        target.update_face_cache()

        # 3. 변경된 정보를 전역 아이템 매니저에 업데이트( 이전에는 copy한 아이템으로 테스트한거였음, 확정된 걸로 교체)
        global_item_manager.update(target._id, target)

        self._add_item(target._id)
        
        # 5. update_top_face
        # _occupancy
        get_top_face_occupancy(self, target)

    def remove(self, target: int | Item):
        """
        bin에서 아이템을 제거합니다.
        item정보의 기준은 무.조.건. global_item_manager에 등록된 값입니다!
        🟠bin에서 제거하기 전 반드시 global_item_manager 정보를 확인하세요!🟠
        """
        item_id = target if isinstance(target, int) else target._id
        self._remove_item(item_id)

    def clear(self):
        self.item_ids.clear()
        p_xy = index.Property()
        p_xz = index.Property()
        p_yz = index.Property()
        self.index_xy = index.Index(properties=p_xy)
        self.index_xz = index.Index(properties=p_xz)
        self.index_yz = index.Index(properties=p_yz)
        self.xy_map.clear()
        self.xz_map.clear()
        self.yz_map.clear()
        self.graph.clear()

    def get_origin_item(self):
        """
        bin에 등록된 아이템 중, origin_item_id에 해당하는 아이템을 반환합니다.
        """
        if self.origin_item_id is None:
            return None
        return global_item_manager.get(self.origin_item_id)

    # ------------------------------------------------------------------
    # 3) 프로퍼티 & 기본 정보
    # ------------------------------------------------------------------
    def getTotalWeight(self) -> float: return self._total_weight()
    def getDensity    (self) -> float: return self.getTotalWeight() / self.volume
    def getTotalVolume(self) -> float: return self._total_volume()

    # ------------------------------------------------------------------
    # 4) 아이템 관리
    # ------------------------------------------------------------------
    # 4‑1) 아이템 추가
    def _add_item(self, iid: int):
        """
        bin에 아이템을 저장합니다.
        item정보의 기준은 무.조.건. global_item_manager에 등록된 값입니다!
        🟠bin에 저장하기 전 반드시 global_item_manager 정보를 확인하세요!🟠
        """
        if iid in self.item_ids:  # 이미 있음
            return
        item = global_item_manager.get(iid)
        if item is None:
            raise ValueError(f"Item id {iid} not found in global_item_manager")

        # origin 설정 (최초 아이템)
        if self.origin_item_id is None:
            self.origin_item_id = iid

        # 리스트에 추가
        self.item_ids.append(iid)

        # R-tree 삽입
        self._insert_rtree(iid, item)

        # 그래프 갱신
        self._update_graph_for_item(iid, item)


    # 4‑2) 아이템 삭제
    def _remove_item(self, iid: int):
        if iid not in self.item_ids:
            return
        # self.origin_item_id가 iid인 경우 None으로 설정
        if self.origin_item_id == iid:
            self.origin_item_id = None
        self.item_ids.remove(iid)
        self._delete_rtree(iid)
        self._remove_from_graph(iid)

    # 4-3) 아이템 업데이트
    def update_item_data(self, item_id: int):
        """
        입력된 item_id에 해당하는 아이템의 위치나 크기 등의 정보가
        변경되었을 때, bin에 저장된 아이템 정보를 업데이트합니다.
        업데이트되는 item정보의 기준은 무.조.건. global_item_manager에 등록된 값입니다!
        🟠업데이트 전 반드시 global_item_manager 정보를 확인하세요!🟠
        """
        # Bin 이 보유한 아이템이 아닌 경우 무시
        if item_id not in self.item_ids:
            return

        item = global_item_manager.get(item_id)
        if item is None:
            return

        # 1) 기존 rtree, map에서 삭제
        self._delete_rtree(item_id)

        # 2) 그래프에서 기존 엣지 삭제
        self._remove_from_graph(item_id)

        # 3) 새 박스 삽입
        self._insert_rtree(item_id, item)

        # 4) 그래프 노드 재계산
        self._update_graph_for_item(item_id, item) # 새 엣지 추가

    # ------------------------------------------------------------------
    # 5) R‑tree 
    # ------------------------------------------------------------------
    # 5-1) R‑tree 삽입
    def _insert_rtree(self, iid: int, item: Item):

        x, y, z  = item.b_position
        ex, ey, ez = item.ex, item.ey, item.ez
        xy = (float(x),  float(y),  float(ex), float(ey))
        xz = (float(x),  float(z),  float(ex), float(ez))
        yz = (float(y),  float(z),  float(ey), float(ez))
        self.index_xy.insert(iid, xy); self.xy_map[iid] = xy
        self.index_xz.insert(iid, xz); self.xz_map[iid] = xz
        self.index_yz.insert(iid, yz); self.yz_map[iid] = yz

    # 5-2) R‑tree 삭제
    def _delete_rtree(self, iid: int):
        if iid in self.xy_map:
            self.index_xy.delete(iid, self.xy_map[iid])
            del self.xy_map[iid]
        if iid in self.xz_map:
            self.index_xz.delete(iid, self.xz_map[iid])
            del self.xz_map[iid]
        if iid in self.yz_map:
            self.index_yz.delete(iid, self.yz_map[iid])
            del self.yz_map[iid]


    # ------------------------------------------------------------------
    # 6) R‑tree 검색 헬퍼
    # ------------------------------------------------------------------
    def search_z(self, z_min, z_max):
        candidate_ids = self.search_xz(BIG_NEG, BIG_POS, z_min, z_max)
        return list(candidate_ids)
    
    def search_xy(self, x_min: float, x_max: float, y_min: float, y_max: float):
        return list(self.index_xy.intersection((float(x_min), float(y_min), float(x_max), float(y_max))))

    def search_xz(self, x_min: float, x_max: float, z_min: float, z_max: float):
        return list(self.index_xz.intersection((float(x_min), float(z_min), float(x_max), float(z_max))))

    def search_yz(self, y_min: float, y_max: float, z_min: float, z_max: float):
        return list(self.index_yz.intersection((float(y_min), float(z_min), float(y_max), float(z_max))))

    def search_xyz(self, x_min, x_max, y_min, y_max, z_min, z_max):
        return set(self.search_xy(x_min,x_max,y_min,y_max)) & \
               set(self.search_xz(x_min,x_max,z_min,z_max)) & \
               set(self.search_yz(y_min,y_max,z_min,z_max))


    # ------------------------------------------------------------------
    # 7) 인접 그래프 관리
    # ------------------------------------------------------------------
    # 7-1) 그래프 삭제
    def _remove_from_graph(self, iid: int):
        if iid in self.graph:
            del self.graph[iid]
        for node in self.graph.values():
            for d in node:
                node[d].discard(iid)

    # 7-2) 그래프 업데이트
    def _update_graph_for_item(self, iid: int, item: Item):
        """
        인접 그래프 노드를 item정보를 기준으로 업데이트합니다.
        🟠아이템의 vertices 정보가 업데이트 되어 있어야 합니다.🟠
        """
        if item is None:
            return

        # 노드 초기화
        if iid not in self.graph:
            self.graph[iid] = {d: set() for d in ['left','right','front','back','top','bottom']}

        # --------------------------------------------------------------
        # (A) 좌우(left/right) 인접 계산
        # --------------------------------------------------------------
        # A.right → 후보들의 left
        _, b_right = item.getFaceInfo('right');  a_rx = b_right['x'][0]
        y0, y1 = b_right['y']; z0, z1 = b_right['z']
        cand = self.search_xyz(a_rx, a_rx + self.margin_x + EPS, y0, y1, z0, z1)
        for oid in cand:
            if oid == iid: continue
            oitem = global_item_manager.get(oid)
            if oitem is None : continue
            _, l_bounds = oitem.getFaceInfo('left');  b_lx = l_bounds['x'][0]
            if abs(b_lx - a_rx) <= self.margin_x and \
               overlap_intervals(y0, y1, l_bounds['y'][0], l_bounds['y'][1]) and \
               overlap_intervals(z0, z1, l_bounds['z'][0], l_bounds['z'][1]):
                self._link_edge(iid, oid, 'right', 'left')

        # A.left  → 후보들의 right
        _, b_left = item.getFaceInfo('left');   a_lx = b_left['x'][0]
        y0, y1 = b_left['y']; z0, z1 = b_left['z']
        cand = self.search_xyz(a_lx - self.margin_x, a_lx, y0, y1, z0, z1)
        for oid in cand:
            if oid == iid: continue
            oitem = global_item_manager.get(oid)
            if oitem is None : continue
            _, r_bounds = oitem.getFaceInfo('right');  b_rx = r_bounds['x'][0]
            if abs(a_lx - b_rx) <= self.margin_x and \
               overlap_intervals(y0, y1, r_bounds['y'][0], r_bounds['y'][1]) and \
               overlap_intervals(z0, z1, r_bounds['z'][0], r_bounds['z'][1]):
                self._link_edge(iid, oid, 'left', 'right')

        # --------------------------------------------------------------
        # (B) 앞/뒤(front/back) 인접 계산
        # --------------------------------------------------------------
        # A.front → 후보들의 back
        _, f_bounds = item.getFaceInfo('front'); a_fy = f_bounds['y'][0]
        x0,x1 = f_bounds['x']; z0,z1 = f_bounds['z']
        cand = self.search_xyz(x0, x1, a_fy - self.margin_y, a_fy, z0, z1)
        for oid in cand:
            if oid == iid: continue
            oitem = global_item_manager.get(oid)
            if oitem is None : continue
            _, b_bounds = oitem.getFaceInfo('back');  b_by = b_bounds['y'][0]
            if abs(b_by - a_fy) <= self.margin_y and \
               overlap_intervals(x0, x1, b_bounds['x'][0], b_bounds['x'][1]) and \
               overlap_intervals(z0, z1, b_bounds['z'][0], b_bounds['z'][1]):
                self._link_edge(iid, oid, 'front', 'back')

        # A.back → 후보들의 front
        _, b_bounds = item.getFaceInfo('back'); a_by = b_bounds['y'][0]
        x0,x1 = b_bounds['x']; z0,z1 = b_bounds['z']
        cand = self.search_xyz(x0, x1, a_by, a_by + self.margin_y, z0, z1)
        for oid in cand:
            if oid == iid: continue
            oitem = global_item_manager.get(oid)
            if oitem is None : continue
            _, f_bounds2 = oitem.getFaceInfo('front');  b_fy = f_bounds2['y'][0]
            if abs(a_by - b_fy) <= self.margin_y and \
               overlap_intervals(x0, x1, f_bounds2['x'][0], f_bounds2['x'][1]) and \
               overlap_intervals(z0, z1, f_bounds2['z'][0], f_bounds2['z'][1]):
                self._link_edge(iid, oid, 'back', 'front')

        # --------------------------------------------------------------
        # (C) 위/아래(top/bottom) 인접 계산
        # --------------------------------------------------------------
        x, y, z  = item.b_position; ex,ey,ez = item.ex,item.ey,item.ez
        # 위
        cand = self.search_xyz(x, ex, y, ey, ez, ez + EPS)
        for oid in cand:
            if oid == iid: continue
            oitem = global_item_manager.get(oid)
            if oitem is None : continue
            oz = oitem.b_position[2]
            if abs(ez - oz) < EPS and \
               overlap_intervals(x, ex, oitem.b_position[0], oitem.ex) and \
               overlap_intervals(y, ey, oitem.b_position[1], oitem.ey):
                self._link_edge(iid, oid, 'top', 'bottom')
        # 아래
        cand = self.search_xyz(x, ex, y, ey, z - EPS, z)
        for oid in cand:
            if oid == iid: continue
            oitem = global_item_manager.get(oid)
            if oitem is None : continue
            oez = oitem.ez
            if abs(oez - z) < EPS and \
               overlap_intervals(x, ex, oitem.b_position[0], oitem.ex) and \
               overlap_intervals(y, ey, oitem.b_position[1], oitem.ey):
                self._link_edge(iid, oid, 'bottom', 'top')

    # 7-3) 그래프 엣지 연결 헬퍼
    def _link_edge(self, a: int, b: int, dir_a: str, dir_b: str):
        for node, d, peer, pd in ((a, dir_a, b, dir_b), (b, dir_b, a, dir_a)):
            if node not in self.graph:
                self.graph[node] = {k: set() for k in ['left','right','front','back','top','bottom']}
            self.graph[node][d].add(peer)

    # ------------------------------------------------------------------
    #  8) 사용자 함수
    # ------------------------------------------------------------------
    def get_all_items(self):
        items = []
        for item_id in self.item_ids:
            item = global_item_manager.get(item_id)
            if item is not None:
                items.append(item)
        return items

    def find_attached_items(self):
        attached_items = []
        for item_id in self.item_ids:
            item = global_item_manager.get(item_id)
            if item.options.get('is_attached', False):
                attached_items.append(item)
        return attached_items

    def highest_ez_fast(self) -> float | None:
        """
        self.index_xz.bounds 를 이용해 최고 ez 를 O(1)로 반환.
        아이템이 없으면 None.
        """
        b = self.index_xz.bounds          # → (min_x, min_z, max_x, max_z) 또는 None
        return None if b is None else b[3]

    
    def group_by_z(self):
        z_dict = {}
        for item_id in self.item_ids:
            item = global_item_manager.get(item_id)
            if item is None:
                continue
            z_val = item.b_position[2]
            if z_val not in z_dict:
                z_dict[z_val] = []
            z_dict[z_val].append(item)
        return z_dict

    def find_item_ids_by_top_z_bst(self, target_z: float, epsilon: float = 1e-6):
        """
        self._item_ids(=아이템 ID 목록)에서 각 아이템의 ez(윗면 z)를
        정렬-이진탐색(bisect)으로 빠르게 찾는다.

        Parameters
        ----------
        target_z : float
            기준이 되는 z 값
        epsilon  : float, optional
            허용 오차(기본 1e-6)

        Returns
        -------
        List[int]  -  조건을 만족하는 item_id 들
        """
        # ① (ez, item_id) 쌍을 만들고 ez 기준 정렬
        ez_pairs = []
        for iid in self._item_ids:
            itm = global_item_manager.get(iid)
            if itm is not None:
                ez_pairs.append((float(itm.ez), iid))

        ez_pairs.sort(key=lambda x: x[0])           # ez 오름차순

        # ② bisect 로 구간(left_idx, right_idx) 구하기
        ez_vals = [p[0] for p in ez_pairs]
        min_z, max_z = target_z - epsilon, target_z + epsilon
        left_idx  = bisect_left(ez_vals, min_z)
        right_idx = bisect_right(ez_vals, max_z)

        # ③ 구간 안의 item_id 추출
        return [ez_pairs[i][1] for i in range(left_idx, right_idx)]
    
        # ------------------------------------------------------------------
    # 9) Dead-space metrics (AFV / DeadVolume)
    # ------------------------------------------------------------------
    def _collect_top_rects_topdown(self):
        """
        위에서 본 XY 투영 사각형들과 각 사각형이 '막는' z_top을 모은다.
        반환: List[Tuple[x0, x1, y0, y1, z_top]]
          - 바닥 베이스: (0..width, 0..height) @ z_top=0
          - 각 아이템 top: (x0..x1, y0..y1) @ z_top=item.ez
        """
        W = float(self.width)   # x
        H = float(self.height)  # y (주의: Bin에서 height=Y)
        D = float(self.depth)   # z

        rects = []
        # 바닥(접근 경로 베이스)
        rects.append((0.0, W, 0.0, H, 0.0))

        for iid in self.item_ids:
            it = global_item_manager.get(iid)
            if it is None:
                continue
            x0, y0, z0 = map(float, it.b_position)
            w, h, d    = map(float, it.getDimension())  # (x=width, y=height, z=depth)
            x1, y1     = x0 + w, y0 + h
            z_top      = z0 + d

            # bin 경계 클램프
            x0c = max(0.0, min(W, x0)); x1c = max(0.0, min(W, x1))
            y0c = max(0.0, min(H, y0)); y1c = max(0.0, min(H, y1))
            ztc = max(0.0, min(D, z_top))
            if x1c > x0c and y1c > y0c:
                rects.append((x0c, x1c, y0c, y1c, ztc))

        return rects

    @staticmethod
    def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> bool:
        """구간 [a1,a2]와 [b1,b2]가 양의 길이로 겹치면 True."""
        return not (a2 <= b1 or b2 <= a1)

    def get_AFV(self) -> float:
        """
        AFV(Accessible Free Volume, 위에서 접근 가능한 자유부피) 계산.
        아이디어:
          1) XY 평면에 바닥/아이템 top 사각형의 모든 엣지를 모아 격자 셀 생성
          2) 각 셀에서 z_top의 최대값을 구함
          3) 셀 부피 = (depth - z_max) * cell_area 를 합산
        반환: float (mm^3)
        """
        W = float(self.width)   # x
        H = float(self.height)  # y
        D = float(self.depth)   # z
        if W <= 0 or H <= 0 or D <= 0:
            return 0.0

        rects = self._collect_top_rects_topdown()
        # 엣지 수집
        xs = {0.0, W}
        ys = {0.0, H}
        for x0, x1, y0, y1, _ in rects:
            xs.add(float(x0)); xs.add(float(x1))
            ys.add(float(y0)); ys.add(float(y1))
        xs = sorted(xs)
        ys = sorted(ys)

        AFV = 0.0
        for i in range(len(xs) - 1):
            cx0, cx1 = xs[i], xs[i+1]
            dx = cx1 - cx0
            if dx <= 0:
                continue
            for j in range(len(ys) - 1):
                cy0, cy1 = ys[j], ys[j+1]
                dy = cy1 - cy0
                if dy <= 0:
                    continue

                # 이 셀의 z_top 최대값
                zmax = 0.0
                for rx0, rx1, ry0, ry1, rz in rects:
                    if self._overlap_1d(cx0, cx1, rx0, rx1) and self._overlap_1d(cy0, cy1, ry0, ry1):
                        if rz > zmax:
                            zmax = rz

                AFV += max(0.0, D - zmax) * dx * dy

        return float(AFV)
    
    def get_AFV_ratio(self) -> float:
        """
        AFV 비율 = AFV / volume
        반환: float (0.0 ~ 1.0)
        """
        V_bin = float(self.volume)
        if V_bin <= 0.0:
            return 0.0
        AFV   = float(self.get_AFV())
        return float(AFV / V_bin)

    def get_deadVolume(self) -> float:
        """
        Dead Volume = (총 여유부피) - AFV
                    = (volume - placed_volume) - AFV
        반환: float (mm^3)
        """
        V_bin    = float(self.volume)
        V_items  = float(self.getTotalVolume())
        AFV      = float(self.get_AFV())
        free_tot = max(0.0, V_bin - V_items)
        dead     = max(0.0, free_tot - AFV)
        return float(dead)
    
    def get_deadVolume_ratio(self) -> float:
        """
        Dead Volume 비율 = Dead Volume / volume
        반환: float (0.0 ~ 1.0)
        """
        V_bin = float(self.volume)
        if V_bin <= 0.0:
            return 0.0
        dead  = float(self.get_deadVolume())
        return float(dead / V_bin)

    
    # ──────────────────────────────────────────────────────────────
    # Bin 내 추가(또는 대체) 메서드
    # ──────────────────────────────────────────────────────────────
    def get_visible_items_topdown(self):
        """
        ▸ z-축(위)에서 투영했을 때 *조금이라도* 노출돼 있는 아이템들의 id
        (또는 Item) 목록을 돌려준다.
        ▸ 동작
            1.  ez(윗면 z) 높은 순으로 정렬
            2.  이미 ‘가려진 면적’(사각형) 집합을 R-tree에 넣어 두고
                ▸ 내 XY 직사각형을 **완전히** 덮는 사각형이 하나라도
                있으면 → 숨김 처리
                ▸ 아니면   → ‘보이는 물건’으로 등록 + 내 사각형을 R-tree에 추가
                (부분만 겹치면 “보인다”로 간주)
        """
        # ① ez 높은 → 낮은 순
        sorted_ids = sorted(
            (iid for iid in self.item_ids             # ← None-item 제외
            if global_item_manager.get(iid) is not None),
            key=lambda iid: global_item_manager.get(iid).ez,
            reverse=True
        )


        visible: list[int] = []          # 결과
        cov_idx = index.Index()          # 이미 “덮여-있는” XY 영역
        rect_map: dict[int, tuple[float,float,float,float]] = {}
        _rid = count()                   # R-tree 내부 id 생성기

        for iid in sorted_ids:
            it   = global_item_manager.get(iid)
            x0,y0   = it.b_position[:2]
            it_w, it_h, _ = it.getDimension()
            x1,y1   = x0 + it_w, y0 + it_h
            my_rect = (float(x0), float(y0), float(x1), float(y1))

            # ② 내 사각형을 완전히 포함하는 덮개(rect)가 있는지?
            hidden = False
            for cid in cov_idx.intersection(my_rect):
                rx0,ry0,rx1,ry1 = rect_map[cid]
                if rx0 <= x0 and ry0 <= y0 and rx1 >= x1 and ry1 >= y1:
                    hidden = True
                    break                        # 하나라도 완전-포함 ⇒ 가려짐

            if not hidden:                       # 조금이라도 보이면
                visible.append(iid)
                rid = next(_rid)
                cov_idx.insert(rid, my_rect)     # ‘가려진 면적’ 갱신
                rect_map[rid] = my_rect

        return [global_item_manager.get(i) for i in visible]


    def get_items_above(self, loaded_item):
        """
        loaded_item 위(z축)에 위치하고, x-y 평면에서 겹치는 아이템들만 반환.
        🟠vertices정보를 쓰기 때문에 loaded_item의 vertices정보가 정확해야함🟠
        """
        # 위에 물건 리스트를 받아오기 위해서 위 공간 범위 정의
        loaded_item.update_face_cache()
        _, top_bounds = loaded_item.getFaceInfo('top')
        
        # 해당 공간에 포함되는 리스트가 있는지 반환
        top_ids_list = self.search_xyz(top_bounds['x'][0], top_bounds['x'][1], top_bounds['y'][0], top_bounds['y'][1], top_bounds['z'][1], self.depth)
        top_list = []
        for top_id in top_ids_list:
            if global_item_manager.has_id(top_id) is False:
                continue
            top_list.append(global_item_manager.get(top_id))
        return top_list

    
    def get_bottom_items_in_graph(self, item):
        """
        주어진 bin 내에서, 특정 item의 그래프 노드를 활용하여,
        해당 아이템의 'bottom' 방향(아래쪽)에 인접한 아이템 객체들을 반환합니다.
        
        Parameters:
        - bin: 현재 사용 중인 bin 객체 (bin.graph에 접근 가능해야 함)
        - item: 기준 아이템 (Item 객체)
        
        Returns:
        - List of Item objects that are attached to the bottom of the given item.
            만약 해당 item에 대해 그래프 노드가 없거나, 'bottom' 연결이 없다면 빈 리스트를 반환.
        """
        if item._id not in self.graph:
            return []
        
        bottom_ids = self.graph[item._id].get('bottom', set())
        bottom_items = []
        for bid in bottom_ids:
            bottom_item = global_item_manager.get(bid)
            if bottom_item is not None:
                bottom_items.append(bottom_item)
        return bottom_items

    
    def get_bottom_items(self, loaded_item):
        """
        loaded_item 아래(z축)에 위치하고, x-y 평면에서 겹치는 아이템들만 반환.
        """        
        # 아래에 물건 리스트를 받아오기 위해서 아래 공간 범위 정의
        loaded_item.update_face_cache()
        _, bottom_bounds = loaded_item.getFaceInfo('bottom')
        
        # 해당 공간에 포함되는 리스트가 있는지 반환
        bottom_ids_list = self.search_xyz(bottom_bounds['x'][0], bottom_bounds['x'][1], bottom_bounds['y'][0], bottom_bounds['y'][1], bottom_bounds['z'][0] - 5, bottom_bounds['z'][0])
        bottom_list = []
        for bottom_id in bottom_ids_list:
            if global_item_manager.has_id(bottom_id) is False:
                continue
            bottom_list.append(global_item_manager.get(bottom_id))
        return bottom_list
    
    def is_top_empty(self, item):
        """
        loaded_item 위쪽에 겹치는 아이템이 하나라도 있으면 True, 아니면 False 반환.
        bin.binIndex를 이용하여 효율적으로 탐색.
        """

        tb_z_min = min(item.ez + EPS, self.depth)
        tb_x_min = max(0, item.b_position[0] + EPS)
        tb_x_max = max(0, item.ex - EPS)
        tb_y_min = max(0, item.b_position[1] + EPS)
        tb_y_max = max(0, item.ey - EPS)
        
        top_list = self.search_xyz(tb_x_min, tb_x_max, tb_y_min, tb_y_max,  tb_z_min, self.depth)

        if len(top_list) == 0:
            return True
        
        top_item = global_item_manager.get(list(top_list)[0])
        if item.name == "gripper" and len(top_list) == 1 and top_item.options['is_attached'] is True:
            return True

        return False


    def get_max_item_size(
        self,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float, float]:
        """
        (x, y, z) 위치에 바닥 모서리를 두고
        ➜  `+x`, `+y`, `+z` 방향으로 뻗을 수 있는
            “최대(가로 w, 세로 h, 높이 d)”를 계산한다.

        Parameters
        ----------
        x, y, z : float
            아이템을 놓을 기준 좌표 (mm)
        Returns
        -------
        (w_max, h_max, d_max) : Tuple[float, float, float]
            주어진 자리에서 놓을 수 있는 최대 치수(mm).
            값이 0 이면 해당 방향으로는 전혀 배치할 수 없다는 의미.
        """
        cx = self.margin_x
        cy = self.margin_y
        cz = 0.0

        # 1) 우선 ‘컨테이너 경계’가 만들어 주는 최댓값
        w_max = max(0.0, self.width  - x)
        h_max = max(0.0, self.height - y)
        d_max = max(0.0, self.depth  - z)

        # 2) 이미 들어-있는 모든 아이템을 살펴보며 각 축별로 갱신
        for iid in self.item_ids:
            itm = global_item_manager.get(iid)
            if itm is None:
                continue

            ix, iy, iz = itm.b_position
            ex, ey, ez = itm.ex, itm.ey, itm.ez

            # ── (a)  +x 방향 ─────────────────────────────────────────
            if overlap_intervals(y, y + h_max, iy - cy, ey + cy) and \
               overlap_intervals(z, z + d_max, iz - cz, ez + cz):

                # 아이템의 *왼쪽* 면이 (x, y, z) 기준 오른쪽에 있다면 폭 후보
                if ix - cx >= x:
                    w_max = min(w_max, ix - cx - x)

            # ── (b)  +y 방향 ─────────────────────────────────────────
            if overlap_intervals(x, x + w_max, ix - cx, ex + cx) and \
               overlap_intervals(z, z + d_max, iz - cz, ez + cz):

                if iy - cy >= y:
                    h_max = min(h_max, iy - cy - y)

            # ── (c)  +z 방향 ─────────────────────────────────────────
            if overlap_intervals(x, x + w_max, ix - cx, ex + cx) and \
               overlap_intervals(y, y + h_max, iy - cy, ey + cy):

                if iz - cz >= z:
                    d_max = min(d_max, iz - cz - z)

        # 음수로 떨어지는 경우(겹쳐서 못 들어가는 경우) 0 으로 클램프
        w_max = max(0.0, w_max)
        h_max = max(0.0, h_max)
        d_max = max(0.0, d_max)
        return w_max, h_max, d_max

    # ------------------------------------------------------------------
    # 10) 기하학 계산 (면 캐시 등)
    # ------------------------------------------------------------------
    def getVertices(self):
        return [
            [0, 0, 0],
            [self.width, 0, 0],
            [self.width, self.height, 0],
            [0, self.height, 0],
            [0, 0, self.depth],
            [self.width, 0, self.depth],
            [self.width, self.height, self.depth],
            [0, self.height, self.depth],
        ]

    def update_face_cache(self):
        direction_map = {
            'left'  : [0, 3, 7, 4],
            'right' : [1, 2, 6, 5],
            'front' : [0, 1, 5, 4],
            'back'  : [3, 2, 6, 7],
            'bottom': [0, 1, 2, 3],
            'top'   : [4, 5, 6, 7],
        }
        self._face_info.clear()
        V = self.getVertices()

        for d, idxs in direction_map.items():
            verts = [V[i] for i in idxs]

            # ── 1) 단위 법선 벡터 계산 ─────────────────────────────
            v1 = np.subtract(verts[1], verts[0], dtype=float)
            v2 = np.subtract(verts[2], verts[0], dtype=float)
            n  = np.cross(v1, v2)            # 이미 float64
            n /= np.linalg.norm(n)           # OK

            # ── 2) 평면 방정식 (a,b,c,d) ──────────────────────────
            a, b, c = n
            d_coef  = -np.dot(n, verts[0])         # d = -n·p₀
            plane   = (a, b, c, d_coef)

            # ── 3) bounds 그대로 ─────────────────────────────────
            xs, ys, zs = zip(*verts)
            bounds = {'x': (min(xs), max(xs)),
                    'y': (min(ys), max(ys)),
                    'z': (min(zs), max(zs))}

            self._face_info[d] = (plane, bounds)

        return self._face_info

    def getFaceInfo(self, direction: str):
        if not self._face_info:
            self.update_face_cache()
        return self._face_info[direction]

    # ------------------------------------------------------------------
    # 11) 시각화 헬퍼 (PainterPlot)
    # ------------------------------------------------------------------
    def render(self, *, show: bool = False, save_path: str = "planning/renders",
            size_annotation: bool = False,
            return_array: bool = False,
            name: str | None = None, alpha: float = .2, fontsize: int = 8,
            save: bool = True, write_num: bool = True, pivots=None, it_ids=None,
            view_elev: float = 60, view_azim: float = -30, view_roll=None, view_dist=None,
            proj_type: str = "persp", topdown: bool = False):
        # 🔽 topdown 요청이면 카메라를 천정 뷰로 강제
        if topdown:
            view_elev, view_azim, proj_type = 90, -90, "ortho"

        painter = PainterPlot(self)
        return painter.plotBoxAndItems(
            title=f"result_{self.name}_{name}",
            size_annotation=size_annotation,
            return_array=return_array,
            alpha=alpha, fontsize=fontsize,
            save=save, show=show, write_num=write_num,
            save_path=save_path, pivots=pivots, it_ids=it_ids,
            # 🔽 반드시 전달!
            view_elev=view_elev, view_azim=view_azim, view_roll=view_roll,
            view_dist=view_dist, proj_type=proj_type
        )

    # ------------------------------------------------------------------
    # 12) 문자열 표현
    # ------------------------------------------------------------------
    def __str__(self):
        attrs = [
            f"partno={self.partno}", f"name={self.name}",
            f"WHD=({self.width},{self.height},{self.depth})",
            f"items={self.number_of_items}",
        ]
        return f"Bin({', '.join(attrs)})"
    
    # ------------------------------------------------------------------
    # 13) 디버깅 헬퍼
    # ------------------------------------------------------------------
    def debug_xy(self):
        all_item_ids = list(self.index_xy.intersection((BIG_NEG, BIG_NEG, BIG_POS, BIG_POS)))
        print("XY index has item IDs:", all_item_ids)
    
    def debug_xz(self):
        all_item_ids = list(self.index_xz.intersection((BIG_NEG, BIG_NEG, BIG_POS, BIG_POS)))
        print("XZ index has item IDs:", all_item_ids)

    def debug_yz(self):
        all_item_ids = list(self.index_yz.intersection((BIG_NEG, BIG_NEG, BIG_POS, BIG_POS)))
        print("YZ index has item IDs:", all_item_ids)

    
    # ────────────────────────────────────────────────
    # 15) 연산-최적화 함수 :
    # ────────────────────────────────────────────────
    def create_composite_item(self, children, base_w, base_h, stacked_d, parent_id=None):
        # 새로 만들 컴포지트 아이템
        comp = Item(
            partno   = "composite",
            name     = f"{children[0].name}_Comp." if children else "Comp.",
            objshape = "cube",
            width    = base_w,
            height   = base_h,
            depth    = stacked_d,
            weight   = sum(ci.weight for ci in children),
            unit     = children[0].unit if children else 'mm',
            parent_id   = None
        )
        comp.is_composite = True

        if parent_id is not None:
            comp.parent_id = parent_id

        # child_items를 ID 기반으로 저장
        comp.children_ids = []
        for c in children:
            c.parent_id = comp._id
            comp.children_ids.append(c._id)

        return comp

    def enroll_composite_item(self, comp: Item, children: list[Item]):
        """
        컴포지트 아이템을 bin에 등록합니다.
        global_item_manager은 도서관. 도서관에 children 정보는 업데이트하고 bin에는 comp 아이템만 저장, 연결되어 children정보는 참조할수있음.
        bin 정보 컴팩트하게 유지.

        🟠 주의! 아이템 정보는 global_item_manager에서 가져오므로 꼭 업데이트 후 사용하세요!🟠
        """
        if comp._id in self.item_ids:
            raise ValueError(f"Composite item with id {comp._id} already exists in the bin.")
        
        global_item_manager.update(comp._id, comp)
        
        # 2) 자식 아이템을 global_item_manager에 업데이트
        for c in children:
            global_item_manager.update(c._id, c)

        # 2) 자식 아이템을 bin에서 삭제
        for cid in comp.children_ids:
                if cid is not None:
                    self.remove(cid)

        # 2) comp.item을 bin에 저장
        self.store(comp)
    
    # ─────────────────────────────────────────────────────────────
    # 16)  연산-최적화 함수: 후처리 병합(post-merge)
    # ─────────────────────────────────────────────────────────────
    def post_merge(self, iid: int) -> int:
        """
        leaf item <iid> 와 하나의 면을 완전히 공유하는
        leaf/composite item 1개를 찾아 Composite 로 병합한다.
        성공하면 1, 조건 미충족이면 0 을 반환한다.
        """
        A = global_item_manager.get(iid)
        if A is None:
            return 0

        # ── 준비: A 의 경계값 & 공통 파라미터 ──────────────────────
        Ax, Ay, Az = map(float, A.b_position)
        Aw, Ah, Ad = map(float, A.getDimension())
        Ax2, Ay2, Az2 = Ax + Aw, Ay + Ah, Az + Ad
        mx, my = self.margin_x + EPS, self.margin_y + EPS

        # ① bin 내 다른 아이템 순회
        for bid in self.item_ids:
            if bid == iid:
                continue
            B = global_item_manager.get(bid)
            if B is None:
                continue
            # 회전이 다르면 패스 (축이 맞아야 네 모서리가 일치 가능)
            if not np.allclose(A.rotation_quat, B.rotation_quat, atol=EPS):
                continue

            Bx, By, Bz = map(float, B.b_position)
            Bw, Bh, Bd = map(float, B.getDimension())
            Bx2, By2, Bz2 = Bx + Bw, By + Bh, Bz + Bd

            # ────────────────────────────────────────────────
            # (1) 좌,우 방향 :  x 축을 따라 붙어 있는지?
            # ────────────────────────────────────────────────
            #   ▸ A.right == B.left  또는  A.left == B.right
            #   ▸ y, z 는 완전히 일치해야 함
            if abs(Ax2 - Bx) <= mx and \
            abs(Ay - By) <= 2.5 + EPS and abs(Ay2 - By2) <= 2.5 + EPS and \
            abs(Az - Bz) <= 2.5 + EPS and abs(Az2 - Bz2) <= 2.5 + EPS:
                # → 좌우 병합
                children      = [A, B]
                base_w        = Aw + Bw
                base_h        = Ah
                stacked_d     = Ad
                base_pos      = [min(Ax, Bx), Ay, Az]
                break

            if abs(Bx2 - Ax) <= mx and \
            abs(Ay - By) <= 2.5 + EPS and abs(Ay2 - By2) <= 2.5 + EPS and \
            abs(Az - Bz) <= 2.5 + EPS and abs(Az2 - Bz2) <= 2.5 + EPS:
                # → 좌우 병합 (반대)
                children      = [B, A]
                base_w        = Aw + Bw
                base_h        = Ah
                stacked_d     = Ad
                base_pos      = [min(Ax, Bx), Ay, Az]
                break

            # ────────────────────────────────────────────────
            # (2) 앞,뒤 방향 :  y 축을 따라 붙어 있는지?
            # ────────────────────────────────────────────────
            if abs(Ay2 - By) <= my and \
            abs(Ax - Bx) <= 2.5 + EPS and abs(Ax2 - Bx2) <= 2.5 + EPS and \
            abs(Az - Bz) <= 2.5 + EPS and abs(Az2 - Bz2) <= 2.5 + EPS:
                # → 앞뒤 병합
                children      = [A, B]
                base_w        = Aw
                base_h        = Ah + Bh
                stacked_d     = Ad
                base_pos      = [Ax, min(Ay, By), Az]
                break

            if abs(By2 - Ay) <= my and \
            abs(Ax - Bx) <= 2.5 + EPS and abs(Ax2 - Bx2) <= 2.5 + EPS and \
            abs(Az - Bz) <= 2.5 + EPS and abs(Az2 - Bz2) <= 2.5 + EPS:
                # → 앞뒤 병합 (반대)
                children      = [B, A]
                base_w        = Aw
                base_h        = Ah + Bh
                stacked_d     = Ad
                base_pos      = [Ax, min(Ay, By), Az]
                break

            # ────────────────────────────────────────────────
            # (3) 상,하 방향 :  z 축을 따라 쌓여 있는지?
            # ────────────────────────────────────────────────
            if abs(Az2 - Bz) <= 2.5 + EPS and \
            abs(Ax - Bx) <= 2.5 + EPS and abs(Ax2 - Bx2) <= 2.5 + EPS and \
            abs(Ay - By) <= 2.5 + EPS and abs(Ay2 - By2) <= 2.5 + EPS:
                # → 상하 병합  (A 아래, B 위)
                children      = [A, B]
                base_w, base_h = Aw, Ah
                stacked_d     = Ad + Bd
                base_pos      = [Ax, Ay, min(Az, Bz)]
                break

            if abs(Bz2 - Az) <= 2.5 + EPS and \
            abs(Ax - Bx) <= 2.5 + EPS and abs(Ax2 - Bx2) <= 2.5 + EPS and \
            abs(Ay - By) <= 2.5 + EPS and abs(Ay2 - By2) <= 2.5 + EPS:
                # → 상하 병합  (B 아래, A 위)
                children      = [B, A]
                base_w, base_h = Aw, Ah
                stacked_d     = Ad + Bd
                base_pos      = [Ax, Ay, min(Az, Bz)]
                break
        else:
            # for-loop 를 다 돌고도 break 못했다 → 병합 대상 없음
            return 0

        # ── 여기까지 오면 children = [A,B] 또는 [B,A] 선정 완료 ──
        comp = self.create_composite_item(children, base_w, base_h, stacked_d)
        comp.b_position      = [round(x, 3) for x in base_pos]
        # children_ids에서 id로 sort
        comp.rotation_quat   = RotationType.RT_WHD   # 회전 동일하므로 아무거나
        comp.update_face_cache()                           # 면 캐시 갱신

        # bin 내 등록 처리 (자식 삭제 + comp 추가)
        self.enroll_composite_item(comp, children)

        return 1

    def simplify(self, max_iter: int = 50) -> int:
        """
        Bin 내부에 적재된 아이템들을 순회하며,
        병합(post_merge) 가능한 것들을 반복적으로 합쳐 상태를 단순화한다.
        
        Returns:
            int: 병합이 발생한 총 횟수
        """
        total_merges = 0
        
        for _ in range(max_iter):
            merged_this_round = False
            # 순회 중 리스트 변경을 방지하기 위해 ID 목록 복사
            current_ids = list(self.item_ids)
            
            for iid in current_ids:
                # post_merge: 성공 시 1, 실패 시 0 반환
                # 성공 시 내부적으로 아이템 삭제/생성이 일어나므로 루프를 break하고 다시 스캔
                if self.post_merge(iid):
                    merged_this_round = True
                    total_merges += 1
                    break
            
            # 더 이상 병합할 게 없으면 종료
            if not merged_this_round:
                break
                
        return total_merges
    
    # ──────────────────────────────────────────────
    #  pickle 지원을 위한 커스텀 state
    # ──────────────────────────────────────────────
    def __getstate__(self):
        """
        • R-tree(Index)는 pickle 불가 → bbox dict만 보존
        • 나머지는 그대로 shallow copy
        """
        state = self.__dict__.copy()
        # ① 인덱스 객체 제거
        for k in ("index_xy", "index_xz", "index_yz"):
            state.pop(k, None)
        # ② R-tree 내용을 유지하기 위해 bbox map만 남긴다 (이미 dict)
        return state

    def __setstate__(self, state):
        """
        언픽클 시:
        ① 빈 Index() 3개를 새로 만들고
        ② 저장해 둔 bbox 로 다시 insert
        """
        self.__dict__.update(state)

        # --- 새 Index 준비 ---
        self.index_xy = index.Index()
        self.index_xz = index.Index()
        self.index_yz = index.Index()

        # --- 저장돼 있던 bbox 삽입 ---
        for iid, rect in self.xy_map.items():
            self.index_xy.insert(iid, rect)
        for iid, rect in self.xz_map.items():
            self.index_xz.insert(iid, rect)
        for iid, rect in self.yz_map.items():
            self.index_yz.insert(iid, rect)
            
    __repr__ = __str__


if __name__ == "__main__":
    """
    • Bin 에 아이템 2개를 넣고
      index_xy 안에 들어 있는 id 들이 같은지 확인
    """
    from utils.constants import BIG_NEG, BIG_POS
    from planning.item import Item

    # (0) 간단히 global_item_manager 정리 (테스트용)
    from planning.itemManager import global_item_manager

    # (1) Bin ----------------------------------------------------------
    bin0 = Bin(width=500, height=400, depth=300, name="orig")

    # (2) Item 두 개 만들어 저장 --------------------------------------
    it1 = Item(width=100, height=100, depth=100,
               unit="mm", b_position=[0, 0, 0], name="box-1")
    it2 = Item(width=80,  height=120, depth=90,
               unit="mm", b_position=[120, 0, 0], name="box-2")
    

