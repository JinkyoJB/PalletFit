from planning.heuristics.ZoneFit.Zone import Zone

import copy
import math


class ZoneNode:
    def __init__(self, zone):
        self.zone = zone      # 노드가 담고 있는 Zone 객체
        self.left = None      # 왼쪽 자식
        self.right = None     # 오른쪽 자식
        self.height = 1       # AVL 트리 높이 계산을 위한 필드

class ZoneAVLTree:
    def __init__(self):
        self.root = None

    #########################
    # 유틸 메서드
    #########################
    @property
    def size(self) -> int:
        """
        트리 내 노드(Zone) 개수를 동적으로 계산하여 반환.
        """
        return self._count_nodes(self.root)


    def _count_nodes(self, node: ZoneNode) -> int:
        """
        재귀적으로 노드를 방문하여 개수를 반환
        """
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    
    def get_height(self, node: ZoneNode) -> int:
        if node is None:
            return 0
        return node.height

    def get_balance(self, node: ZoneNode) -> int:
        if node is None:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def update_height(self, node: ZoneNode):
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

    def rotate_right(self, z: ZoneNode) -> ZoneNode:
        y = z.left
        T3 = y.right

        # 회전
        y.right = z
        z.left = T3

        # 높이 갱신
        self.update_height(z)
        self.update_height(y)

        return y

    def rotate_left(self, z: ZoneNode) -> ZoneNode:
        y = z.right
        T2 = y.left

        # 회전
        y.left = z
        z.right = T2

        # 높이 갱신
        self.update_height(z)
        self.update_height(y)

        return y

    #########################
    # Zone 비교용 메서드
    #########################
    def _compare_zones(self, z1, z2):
        """
        leftover가 기준:
          z1.leftover < z2.leftover => -1
          z1.leftover > z2.leftover => +1
          같다면 (x, y, z) 순서대로 비교
        """
        if z1.leftover < z2.leftover:
            return -1
        elif z1.leftover > z2.leftover:
            return 1
        else:
            # leftover 같다면, x → y 순
            if z1.x < z2.x:
                return -1
            elif z1.x > z2.x:
                return 1
            else:
                if z1.y < z2.y:
                    return -1
                elif z1.y > z2.y:
                    return 1
                else:
                    if z1.z < z2.z:
                        return -1
                    elif z1.z > z2.z:
                        return 1
                    else:
                        return 0


    #########################
    # 삽입
    #########################
    def _insert(self, node: ZoneNode, zone) -> ZoneNode:

        # -------------------------
        # 0) Zone의 유효성 검사
        # -------------------------
        if zone.width <= 0 or zone.height <= 0 or zone.depth <= 0:
            return node  # 유효하지 않은 Zone은 삽입하지 않고 그대로 반환
        
        # -------------------------
        # 0.5) 기존 노드들과 3D Overlap 검사
        # -------------------------
        # if self._overlaps_with_any(self.root, zone):
        #     # 하나라도 겹치면, 새 zone 삽입하지 않고 종료
        #     return node

        # -------------------------
        # 1) 기존 AVL 삽입 로직
        # -------------------------
        if node is None:
            return ZoneNode(zone)

        comp_result = self._compare_zones(zone, node.zone)
        if comp_result < 0:
            node.left = self._insert(node.left, zone)
        else:
            # 동일하거나 큰 경우 오른쪽
            node.right = self._insert(node.right, zone)

        # 높이 갱신
        self.update_height(node)
        # 균형도 확인
        balance = self.get_balance(node)

        # Left Left
        if balance > 1 and self._compare_zones(zone, node.left.zone) < 0:
            return self.rotate_right(node)

        # Right Right
        if balance < -1 and self._compare_zones(zone, node.right.zone) > 0:
            return self.rotate_left(node)

        # Left Right
        if balance > 1 and self._compare_zones(zone, node.left.zone) > 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)

        # Right Left
        if balance < -1 and self._compare_zones(zone, node.right.zone) < 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node


    def insert(self, zone):
        self.root = self._insert(self.root, zone)

    #########################
    # 삭제
    #########################
    def _get_min_value_node(self, node: ZoneNode):
        current = node
        while current.left:
            current = current.left
        return current

    def _delete(self, node: ZoneNode, zone) -> ZoneNode:
        if node is None:
            return node

        comp_result = self._compare_zones(zone, node.zone)
        if comp_result < 0:
            node.left = self._delete(node.left, zone)
        elif comp_result > 0:
            node.right = self._delete(node.right, zone)
        else:
            # 노드가 삭제 대상
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                # 양쪽 자식 다 있는 경우 -> 오른쪽 서브트리 최소 노드 치환
                min_node = self._get_min_value_node(node.right)
                node.zone = min_node.zone
                node.right = self._delete(node.right, min_node.zone)

        if node is None:
            return None
        
        self.update_height(node)
        balance = self.get_balance(node)

        # Left Left
        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.rotate_right(node)

        # Left Right
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)

        # Right Right
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.rotate_left(node)

        # Right Left
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node

    def delete(self, zone):
        self.root = self._delete(self.root, zone)

    #########################
    # 중위순회 (leftover 오름차순)
    #########################
    def in_order_traversal(self, node, result_list):
        if node:
            self.in_order_traversal(node.left, result_list)
            result_list.append(node.zone)
            self.in_order_traversal(node.right, result_list)

    def get_sorted_zones_leftover(self):
        """
        leftover(작은->큰), leftover 같으면 (x,y) 순으로 정렬된 Zone 리스트
        """
        result = []
        self.in_order_traversal(self.root, result)
        return result
    


    #########################
    # 두 Zone 병합
    #########################
    @staticmethod
    def merge_two_zones(zone1, zone2):
        """
        zone1, zone2를 AABB(축 정렬 바운딩 박스) 방식으로 병합해 새로운 zone을 반환.
        즉, 두 Zone을 완전히 감싸는 x,y,z 범위를 구해, 그 범위로 하나의 직육면체 Zone 생성.
        """
        # 1) 두 Zone의 min/max 좌표
        new_x1 = min(zone1.x, zone2.x)
        new_y1 = min(zone1.y, zone2.y)
        new_z1 = min(zone1.z, zone2.z)

        new_x2 = max(zone1.x + zone1.width,  zone2.x + zone2.width)
        new_y2 = max(zone1.y + zone1.height, zone2.y + zone2.height)
        new_z2 = max(zone1.z + zone1.depth,  zone2.z + zone2.depth)

        # 2) bounding box 의 크기 계산
        new_width  = new_x2 - new_x1
        new_height = new_y2 - new_y1
        new_depth  = new_z2 - new_z1

        # 3) 새 Zone 생성 (이름, 가중치 등은 필요에 따라 합산/조정)
        new_name = f"mz_{new_x1}_{new_y1}_{new_z1}"
        new_zone = Zone(
            x=new_x1,
            y=new_y1,
            z=new_z1,
            width=new_width,
            height=new_height,
            depth=new_depth,
            direction=0,
            unit=zone1.unit,  # zone1, zone2가 같은 단위라고 가정
            max_weight=zone1.max_weight + zone2.max_weight,
            name=new_name,
        )

        # (option) items 병합
        # new_zone.itemTree = merge_trees_into_new(zone1.itemTree, zone2.itemTree)

        new_zone.margin_x = zone1.margin_x
        new_zone.margin_y = zone1.margin_y

        return new_zone


    def merge_zones(self, zone1, zone2):
        """
        트리에 이미 zone1, zone2가 들어있다고 가정하고,
        둘을 삭제한 뒤 merge_two_zones()로 병합 zone을 만든 다음 다시 삽입
        """
        # 먼저 zone1, zone2를 트리에서 제거
        self.delete(zone1)
        self.delete(zone2)

        # 두 Zone 병합
        merged_zone = self.merge_two_zones(zone1, zone2)

        # 병합된 Zone을 다시 트리에 삽입
        self.insert(merged_zone)

    def printTree(self):
        """
        트리 출력 (디버깅용)
        """
        def _print_tree(node, level):
            if node:
                _print_tree(node.left, level + 1)
                print()
                print(f"{node.zone}")
                print()
                _print_tree(node.right, level + 1)

        _print_tree(self.root, 0)

    
    def get_sorted_zones_leftover_desc(self):
        """
        leftover(큰->작은), leftover가 같으면 (x,y) 순으로 정렬된 Zone 리스트를 반환한다.
        """
        result = []
        # 기존 inorder_traversal 로 모든 node.zone을 수집
        self.in_order_traversal(self.root, result)

        # leftover 내림차순 -> leftover가 같은 경우 x,y 오름차순
        # 예: zone.leftover, zone.x, zone.y 를 사용한다고 가정
        result.sort(key=lambda zone: (-zone.leftover, zone.x, zone.y))

        return result

    
    def find_zone(self, target_zone):
        """
        target_zone과 leftover, x, y가 모두 동일한 Zone을 트리에서 찾아 반환.
        없으면 None을 반환.
        """
        current = self.root
        while current is not None:
            comp = self._compare_zones(target_zone, current.zone)
            if comp == 0:
                # leftover, x, y 모두 같은 경우
                return current.zone
            elif comp < 0:
                current = current.left
            else:
                current = current.right

        return None
    
    def find_available_zones(self, item):
        """
        leftover >= item.volume 이면서,
        rotation_quat 0 또는 1 중 하나라도 
        (zone.width >= item.width and zone.height >= item.height)
        만족하는 zone을 리스트로 반환
        """
        # (1) leftover 충족 zone 찾기
        candidate_zones = self.find_zones_by_leftover_at_least(item.volume)
        
        # (2) rotation=0,1 별로 item width,height
        #    예: rotation=0 -> (w0, h0, _)
        #       rotation=1 -> (w1, h1, _)
        #  실제 item에 따라 'BasicRotation'이 다를 수 있지만,
        #  질문상 rt=0, rt=1 두 가지만 고려한다고 가정
        original_rt = item.rotation_quat  # 임시로 저장
        # 회전 0
        item.rotation_quat = 0
        w0, h0, _ = item.getDimension()
        # 회전 1
        item.rotation_quat = 1
        w1, h1, _ = item.getDimension()
        # rotation 복원
        item.rotation_quat = original_rt

        available_zones = []
        for z in candidate_zones:
            # (3) zone.width, zone.height 로 배치 가능여부 검사
            # 회전0이 가능?  or  회전1이 가능?
            can_fit_rt0 = (z.width >= w0 and z.height >= h0)
            can_fit_rt1 = (z.width >= w1 and z.height >= h1)

            if can_fit_rt0 or can_fit_rt1:
                # 하나라도 만족하면 가능 zone
                available_zones.append(z)

        return available_zones

        
    def find_zones_by_leftover_at_least(self, min_leftover):
        """ leftover >= min_leftover 인 zone들을 리스트로 반환 """
        result = []
        self._collect_by_leftover_at_least(self.root, min_leftover, result)
        return result

    def _collect_by_leftover_at_least(self, node, min_leftover, result):
        if not node:
            return
        # node.zone.leftover 확인
        if node.zone.leftover >= min_leftover:
            # 왼쪽 서브트리에도 leftover가 min_leftover 이상인 노드가 있을 수 있으므로 검색
            self._collect_by_leftover_at_least(node.left, min_leftover, result)
            # 현재 node도 조건 충족 -> 추가
            result.append(node.zone)
            # 오른쪽 서브트리는 당연히 leftover >= min_leftover
            self._collect_by_leftover_at_least(node.right, min_leftover, result)
        else:
            # node.zone.leftover < min_leftover 이면, 왼쪽 서브트리는 더 작을 수 있으니 볼 필요가 없고,
            # 오히려 오른쪽 서브트리는 leftover가 클 수 있음 -> 오른쪽 서브트리만 검색
            self._collect_by_leftover_at_least(node.right, min_leftover, result)

    def find_zones_in_xy_range(self, x_min, x_max, y_min, y_max):
        result = []
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            z = node.zone
            if (z.x >= x_min and z.x + z.width <= x_max and
                z.y >= y_min and z.y + z.height <= y_max):
                result.append(z)
            dfs(node.right)
        dfs(self.root)
        return result
    
    #########################
    # 새로운 메서드: leftover 통계 계산
    #########################
    def get_leftover_status(self):
        """
        트리에 저장된 모든 Zone들의 leftover 값을 기반으로
        - 가장 큰 leftover 값
        - 가장 작은 leftover 값
        - leftover의 평균값
        - max_leftover를 가진 Zone의 3D 대각선 길이
        (대각선 = sqrt(width^2 + height^2 + depth^2))
        
        반환 형식: (max_leftover, min_leftover, average_leftover, diagonal_of_max_leftover)
        """
        if self.root is None:
            # 트리가 비어있다면 leftover 관련 통계가 없으므로 모두 0 처리
            return 0, 0, 0, 0

        min_leftover = math.inf
        max_leftover = -math.inf
        total_leftover = 0
        count = 0

        # 추가: leftover가 최대인 Zone을 저장
        node_with_max_leftover = None

        def traverse(node):
            nonlocal min_leftover, max_leftover, total_leftover, count
            nonlocal node_with_max_leftover

            if node is None:
                return

            traverse(node.left)

            leftover = node.zone.leftover
            # min / max 업데이트
            if leftover < min_leftover:
                min_leftover = leftover
            if leftover > max_leftover:
                max_leftover = leftover
                node_with_max_leftover = node.zone  # 여기서 최대 leftover Zone 갱신

            total_leftover += leftover
            count += 1

            traverse(node.right)

        # 중위순회
        traverse(self.root)

        # 평균 leftover
        average_leftover = total_leftover / count if count > 0 else 0

        # max leftover를 가진 Zone의 대각선 길이
        diagonal_of_max_leftover = 0
        if node_with_max_leftover is not None:
            w = node_with_max_leftover.width
            h = node_with_max_leftover.height
            d = node_with_max_leftover.depth
            # sqrt()는 float을 반환
            diag_float = math.sqrt(float(w*w + h*h + d*d))
            diagonal_of_max_leftover = diag_float

        return max_leftover, min_leftover, average_leftover, diagonal_of_max_leftover
    
    def copy_tree(self):
        """
        현재 ZoneAVLTree의 완전 복사본(deep copy)을 만들어 반환한다.
        원본 트리와 똑같은 구조(AVL 높이, 왼/오른 자식 배치)와
        Zone 객체들을 전부 복제한 새 트리를 얻을 수 있다.
        """

        def _copy_subtree(node):
            if node is None:
                return None

            # 1) node.zone 복사
            new_zone = copy.deepcopy(node.zone)

            # 2) 새 ZoneNode 생성
            new_node = ZoneNode(new_zone)

            # 3) 왼/오른 자식 재귀 복사
            new_node.left = _copy_subtree(node.left)
            new_node.right = _copy_subtree(node.right)

            # 4) 높이 필드도 동일하게 복사
            new_node.height = node.height

            return new_node

        # 새 트리 인스턴스
        new_tree = ZoneAVLTree()
        # 루트 복제
        new_tree.root = _copy_subtree(self.root)

        return new_tree

    def find_zones_that_intersect_item_and_corners(self, item):
        """
        1) 아이템과 '부분적으로라도' 3D 영역이 겹치는(교차하는) Zone을 찾는다.
        2) 각 Zone에 대해:
        - 아이템의 꼭짓점(getVertices()) 중 Zone 내부에 들어 있는 점들을 수집
        - 만약 꼭짓점이 하나도 없는데도 overlap이면,
            '아이템-Zone 교차영역의 우측상단 모서리'를 반환
        return:
        [
            (zone1, [ [x1,y1,z1], [x2,y2,z2], ... ]),
            (zone2, [ [x3,y3,z3], ... ]),
            ...
        ]
        """

        # 1) 아이템의 꼭짓점 8개 (이미 Item 클래스의 getVertices() 제공)
        item_corners = item.getVertices()
        # 예: [[x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],
        #      [x, y, z+d], ..., [x+w, y+h, z+d]]

        # 2) 아이템의 3D 좌표 범위 (min~max)
        #    (겹침 여부 판단용)
        xs = [v[0] for v in item_corners]
        ys = [v[1] for v in item_corners]
        zs = [v[2] for v in item_corners]
        ix_min, ix_max = min(xs), max(xs)
        iy_min, iy_max = min(ys), max(ys)
        iz_min, iz_max = min(zs), max(zs)

        # 3) 트리에 들어있는 Zone 전부 가져오기
        all_zones = self.get_sorted_zones_leftover()

        results = []
        for z in all_zones:
            # Zone의 min~max
            zx_min, zy_min, zz_min = z.x, z.y, z.z
            zx_max = zx_min + z.width
            zy_max = zy_min + z.height
            zz_max = zz_min + z.depth

            # 3D overlap (부분 겹침) 조건
            overlap_x = (ix_max > zx_min) and (ix_min < zx_max)
            overlap_y = (iy_max > zy_min) and (iy_min < zy_max)
            overlap_z = (iz_max > zz_min) and (iz_min < zz_max)

            if not (overlap_x and overlap_y and overlap_z):
                # 아이템과 전혀 겹치지 않음
                continue

            # (A) 꼭짓점 중 Zone 내부에 들어 있는 점 찾기
            corners_in_zone = []
            for corner in item_corners:
                cx, cy, cz = corner
                if (zx_min <= cx <= zx_max and
                    zy_min <= cy <= zy_max and
                    zz_min <= cz <= zz_max):
                    corners_in_zone.append(corner)

            if corners_in_zone:
                # (A-1) 아이템 꼭짓점 일부가 Zone 안에 있음
                results.append((z, corners_in_zone))
            else:
                # (A-2) 꼭짓점은 없는데 내부(몸통)만 겹침
                # 교차영역의 AABB (intersection box)
                ix_i_min = max(ix_min, zx_min)
                iy_i_min = max(iy_min, zy_min)
                iz_i_min = max(iz_min, zz_min)

                ix_i_max = min(ix_max, zx_max)
                iy_i_max = min(iy_max, zy_max)
                iz_i_max = min(iz_max, zz_max)

                # '우측상단' 모서리를 (ix_i_max, iy_i_max, iz_i_max) 라고 가정
                top_right_corner = [ix_i_max, iy_i_max, iz_i_max]

                results.append((z, [top_right_corner]))

        return results


    def get_sorted_zones_name(self):
        """
        Zone의 name을 기준으로 오름차순 정렬하여 리스트로 반환.
        """
        # 1) 트리에 있는 모든 Zone을 리스트로 수집 (in-order)
        all_zones = []
        self.in_order_traversal(self.root, all_zones)
        # 여기서 all_zones는 leftover 기준 정렬이 아니라,
        # 단순히 '중위순회' 순서로 모인 Zone들의 리스트입니다.

        # 2) zone.name 기준으로 정렬 (문자열 오름차순)
        all_zones.sort(key=lambda z: z.name)

        return all_zones



    def get_sorted_zones_zyx(self):
        """
        Zone의 (z, y, x) 순서로 정렬된 리스트를 반환한다.
        """
        # 1) 트리에 있는 모든 Zone을 리스트로 수집 (in-order)
        all_zones = []
        self.in_order_traversal(self.root, all_zones)

        # 2) (z, y, x) 순서로 정렬
        all_zones.sort(key=lambda z: (z.z, z.y, z.x))

        return all_zones