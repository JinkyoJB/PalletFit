from utils.Pivot import Pivot
from utils.constants import EPS
import numpy as np

def _snap(v):       # → 정수 격자
    return round(float(v)/EPS)

def _snap(v):          # 0.1 mm·deg 이내면 같은 셀
    return round(float(v) / EPS)

def _pivot_key(p):
    #  (z,y,x,w,xq,yq,zq)  7-tuple  모두 정수 격자로!
    return (
        _snap(p.z), _snap(p.y), _snap(p.x),
        *(_snap(v) for v in p.rt)     # 4 요소 모두
    )

def _pivot_key_from_values(x, y, z, rt):
    return (
        _snap(z), _snap(y), _snap(x),
        *(_snap(v) for v in rt)
    )

class PivotNode:
    def __init__(self, pivot):
        self.pivot = pivot
        self.left = None
        self.right = None
        self.height = 1

class PivotAVLTree:
    def __init__(self):
        self.root = None

    #########################
    # 유틸 메서드
    #########################
    @property
    def size(self) -> int:
        """
        트리 내 노드 개수를 반환
        """
        return self._count_nodes(self.root)

    def _count_nodes(self, node: PivotNode) -> int:
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def get_height(self, node: PivotNode) -> int:
        if node is None:
            return 0
        return node.height

    def get_balance(self, node: PivotNode) -> int:
        if node is None:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def update_height(self, node: PivotNode):
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

    # ───────── 안전한 회전 함수 ─────────
    def rotate_right(self, z: PivotNode) -> PivotNode:
        """LL·RL 보정용.  z.left 가 없으면 회전하지 않고 z 반환"""
        if z is None or z.left is None:
            return z
        y  = z.left
        T3 = y.right
        y.right = z
        z.left  = T3
        self.update_height(z)
        self.update_height(y)
        return y

    def rotate_left(self, z: PivotNode) -> PivotNode:
        """RR·LR 보정용.  z.right 가 없으면 회전하지 않고 z 반환"""
        if z is None or z.right is None:
            return z
        y  = z.right
        T2 = y.left
        y.left  = z
        z.right = T2
        self.update_height(z)
        self.update_height(y)
        return y
    # ───────────────────────────────────
    #########################
    # Pivot 비교 (정적 좌표만 사용)
    #########################
    def _compare_pivots(self, p1, p2):
        k1, k2 = _pivot_key(p1), _pivot_key(p2)
        return (k1 > k2) - (k1 < k2)

    def match(self, x, y, z, rt):
        target_key = _pivot_key_from_values(x, y, z, rt)
        node = self.root
        while node:
            k = _pivot_key(node.pivot)
            if k == target_key:
                return node.pivot
            node = node.left if target_key < k else node.right
        return None


    #########################
    # 삽입
    #########################
    def _insert(self, node: PivotNode, pivot: Pivot) -> PivotNode:
        """AVL 삽입 + 보정 (z,y,x,rt 사전식 비교)"""
        if node is None:
            return PivotNode(pivot)

        if node and _pivot_key(pivot) == _pivot_key(node.pivot):
            return node          # 같은 Pivot = 중복 → 삽입 안 함


        comp = self._compare_pivots(pivot, node.pivot)
        if comp < 0:
            node.left  = self._insert(node.left,  pivot)
        else:
            node.right = self._insert(node.right, pivot)

        # ① 높이 갱신
        self.update_height(node)
        balance = self.get_balance(node)          #  L-R

        # ───── 불균형 보정 ─────
        # (1) Left-Left
        if balance > 1 and node.left and \
        self._compare_pivots(pivot, node.left.pivot) < 0:
            return self.rotate_right(node)

        # (2) Right-Right
        if balance < -1 and node.right and \
        self._compare_pivots(pivot, node.right.pivot) > 0:
            return self.rotate_left(node)

        # (3) Left-Right
        if balance > 1 and node.left and \
        self._compare_pivots(pivot, node.left.pivot) > 0:
            node.left = self.rotate_left(node.left)      # node.left 존재 보장
            return self.rotate_right(node)

        # (4) Right-Left
        if balance < -1 and node.right and \
        self._compare_pivots(pivot, node.right.pivot) < 0:
            node.right = self.rotate_right(node.right)   # node.right 존재 보장
            return self.rotate_left(node)

        return node
    
    def insert(self, pivot: Pivot):
        """
        AVL 트리에 새로운 Pivot을 삽입한다.
        (z, y, x, rt)가 동일한 Pivot이 이미 존재하면 삽입하지 않는다.
        """
        # 중복 검사
        if self.match(pivot.x, pivot.y, pivot.z, pivot.rt) is not None:
            # print(f"[insert] 이미 존재하는 Pivot ({pivot.x}, {pivot.y}, {pivot.z}, rt={pivot.rt})")
            return False

        self.root = self._insert(self.root, pivot)
        return True

    #########################
    # 삭제
    #########################
    def _get_min_value_node(self, node: PivotNode):
        current = node
        while current.left:
            current = current.left
        return current

    def _delete(self, node: PivotNode, pivot: Pivot) -> PivotNode:
        if node is None:
            return None

        comp_result = self._compare_pivots(pivot, node.pivot)
        if comp_result < 0:
            node.left = self._delete(node.left, pivot)
        elif comp_result > 0:
            node.right = self._delete(node.right, pivot)
        else:
            # 삭제 대상
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                # 오른쪽 서브트리 최소 노드를 찾아 현재 노드와 교체
                min_node = self._get_min_value_node(node.right)
                node.pivot = min_node.pivot
                node.right = self._delete(node.right, min_node.pivot)

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

    def delete(self, pivot: Pivot):
        self.root = self._delete(self.root, pivot)

    def delete_by_coordinates(self, x, y, z, rt):
        target_pivot = self.match(x, y, z, rt)
        if target_pivot:
            self.delete(target_pivot)
            return True
        return False


    #########################
    # 순회
    #########################
    def in_order_traversal(self):
        yield from self._in_order_traversal(self.root)

    def _in_order_traversal(self, node: PivotNode):
        if node is not None:
            yield from self._in_order_traversal(node.left)
            yield node.pivot
            yield from self._in_order_traversal(node.right)

    def pre_order_traversal(self):
        yield from self._pre_order_traversal(self.root)

    def _pre_order_traversal(self, node: PivotNode):
        if node is not None:
            yield node.pivot
            yield from self._pre_order_traversal(node.left)
            yield from self._pre_order_traversal(node.right)

    def post_order_traversal(self):
        yield from self._post_order_traversal(self.root)

    def _post_order_traversal(self, node: PivotNode):
        if node is not None:
            yield from self._post_order_traversal(node.left)
            yield from self._post_order_traversal(node.right)
            yield node.pivot

    #########################
    # 디버그 출력
    #########################
    def printTree(self):
        self._printTree(self.root)

    def _printTree(self, node: PivotNode):
        if node is not None:
            self._printTree(node.left)
            print(f"Pivot: x={node.pivot.x}, y={node.pivot.y}, z={node.pivot.z}, rt={node.pivot.rt}")
            self._printTree(node.right)
