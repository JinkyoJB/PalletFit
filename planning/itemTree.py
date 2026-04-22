from planning.itemManager import global_item_manager


class itemTreeNode:
    def __init__(self, item_id: int):
        """
        매개변수:
            item_id (int): Item._id 값 (ItemManager.register() 호출되어 할당된 식별자)
        """
        self.item_id = item_id
        self.left = None
        self.right = None
        self.height = 1



class ItemTree:
    def __init__(self):
        self.root = None

    #########################
    # 유틸 메서드
    #########################
    @property
    def size(self) -> int:
        """
        트리에 들어있는 (Composite 아닌) 실제 아이템의 총 개수를 반환.
        """
        return self._count_nodes(self.root)

    def _count_nodes(self, node):
        """
        node (itemTreeNode)를 루트로 하는 서브트리에 있는
        '실제 아이템' 개수를 반환.
        """
        if node is None:
            return 0

        item = global_item_manager.get(node.item_id)
        if item is None:
            # 혹은 raise Error
            return 0
        
        # 이 노드가 composite 아니면 1
        current_count = 0
        if not item.is_composite:
            current_count = 1
        
        # 자식들도 재귀적으로 검사
        current_count += self._count_item_children(item)

        return (current_count 
                + self._count_nodes(node.left) 
                + self._count_nodes(node.right))

    def _count_item_children(self, item):
        """
        'item' 객체의 children_ids(재귀 구조)에서
        Composite가 아닌 아이템을 전부 세어 반환.
        """
        total = 0
        for child_id in item.children_ids:
            child = global_item_manager.get(child_id)
            if child is None:
                continue
            if not child.is_composite:
                total += 1
            # 손자, 증손자...
            total += self._count_item_children(child)
        return total

    def get_height(self, node):
        return node.height if node else 0

    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def update_height(self, node):
        node.height = 1 + max(self.get_height(node.left),
                              self.get_height(node.right))

    def rotate_right(self, z):
        y = z.left
        T3 = y.right

        # 회전
        y.right = z
        z.left = T3

        self.update_height(z)
        self.update_height(y)
        return y

    def rotate_left(self, z):
        y = z.right
        T2 = y.left

        # 회전
        y.left = z
        z.right = T2

        self.update_height(z)
        self.update_height(y)
        return y

    def _compare_item_ids(self, id1, id2):
        item1 = global_item_manager.get(id1)
        item2 = global_item_manager.get(id2)
        if (item1 is None) or (item2 is None):
            return 0

        w1, h1, d1 = item1.getDimension()
        w2, h2, d2 = item2.getDimension()

        x1, y1, z1 = item1.b_position
        x2, y2, z2 = item2.b_position

        pos1 = (z1 + d1, y1 + h1, x1 + w1)
        pos2 = (z2 + d2, y2 + h2, x2 + w2)

        # 💡 Composite 여부 우선 판단 (composite는 항상 뒤로 정렬)
        if item1.is_composite != item2.is_composite:
            return -1 if not item1.is_composite else 1

        # 기존 위치 기반 비교
        if pos1 < pos2:
            return -1
        elif pos1 > pos2:
            return 1
        else:
            # 💡 fallback: ID로 비교해 중복 방지
            return -1 if id1 < id2 else (1 if id1 > id2 else 0)


    #########################
    # 삽입
    #########################
    def _insert(self, node, item_id):
        # 유효성 검사
        item = global_item_manager.get(item_id)
        if (item is None) or (not hasattr(item, 'b_position')):
            raise ValueError("Invalid item_id or item has no 'b_position'.")

        if node is None:
            # 새 노드
            return itemTreeNode(item_id)

        comp = self._compare_item_ids(item_id, node.item_id)
        if comp < 0:
            node.left = self._insert(node.left, item_id)
        elif comp > 0:
            node.right = self._insert(node.right, item_id)
        # else:  # 같은 값 => Duplicate
        #     raise ValueError("Duplicate item not allowed.")

        # 높이, 밸런스 갱신
        self.update_height(node)
        balance = self.get_balance(node)

        # 회전 처리
        # LL
        if balance > 1 and self._compare_item_ids(item_id, node.left.item_id) < 0:
            return self.rotate_right(node)
        # RR
        if balance < -1 and self._compare_item_ids(item_id, node.right.item_id) > 0:
            return self.rotate_left(node)
        # LR
        if balance > 1 and self._compare_item_ids(item_id, node.left.item_id) > 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        # RL
        if balance < -1 and self._compare_item_ids(item_id, node.right.item_id) < 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node

    def insert(self, item):
        """
        아이템(혹은 아이템 ID)을 트리에 삽입
        """
        # item이 실제 Item 객체라면, id만 추출
        if hasattr(item, '_id'):
            item_id = item._id
        else:
            item_id = item  # 혹은 TypeError
        
        self.root = self._insert(self.root, item_id)

    #########################
    # 삭제
    #########################
    def _get_min_value_node(self, node):
        current = node
        while current and current.left:
            current = current.left
        return current

    def _delete(self, node, item_id):
        if node is None:
            raise ValueError("Item not found in the tree.")

        comp = self._compare_item_ids(item_id, node.item_id)
        if comp < 0:
            node.left = self._delete(node.left, item_id)
        elif comp > 0:
            node.right = self._delete(node.right, item_id)
        else:
            # 노드가 삭제 대상
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                # 양쪽 자식 다 있는 경우 -> 오른쪽 서브트리 최소 노드 가져오기
                min_node = self._get_min_value_node(node.right)
                # 현 노드의 item_id를 min_node의 item_id로 교체
                node.item_id = min_node.item_id
                # 오른쪽 서브트리에서 min_node를 삭제
                node.right = self._delete(node.right, min_node.item_id)

        if node is None:
            return node
        
        # 높이 갱신, 밸런스 처리
        self.update_height(node)
        balance = self.get_balance(node)

        # 회전
        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.rotate_right(node)
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.rotate_left(node)
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node

    def delete(self, item):
        """
        item이 최상위(트리에 직접 들어있는)라면, AVLTree에서 삭제하고 True 반환.
        item이 어떤 다른 아이템의 자식이면, 그 부모의 children_ids에서 제거 후 True 반환.
        """
        if item is None:
            return False

        # item이 객체인지, id인지 분기
        if hasattr(item, '_id'):
            item_id = item._id
        else:
            item_id = item

        # (1) item(혹은 id)의 최상위 부모를 찾음
        top_parent_id = self._get_top_parent_id(item_id)
        if top_parent_id is None:
            return False

        if top_parent_id != item_id:
            # => 자식이므로, 부모 객체에서 제거
            parent_obj = global_item_manager.get(top_parent_id)
            child_obj  = global_item_manager.get(item_id)
            if (parent_obj is None) or (child_obj is None):
                return False
            
            if item_id in parent_obj.children_ids:
                parent_obj.children_ids.remove(item_id)
                child_obj.parent_id = None
                return True
            else:
                return False
        else:
            # => 최상위 아이템 -> AVL에서 삭제
            if self.find(item_id) is None:
                return False
            
            # 🔥 자식들 복구
            item = global_item_manager.get(item_id)
            if item and item.is_composite:
                for child_id in item.children_ids:
                    child = global_item_manager.get(child_id)
                    if child:
                        child.parent_id = None
                        self.insert(child)

            self.root = self._delete(self.root, item_id)
            return True

    #########################
    # 검색
    #########################
    def find(self, item):
        """
        item(혹은 id)을 트리에서 검색하여 존재하면 리턴, 없으면 None
        단, 자식까지도 포함(부모가 맞다면 children_ids 재귀로 탐색)
        """
        if item is None:
            return None
        if hasattr(item, '_id'):
            item_id = item._id
        else:
            item_id = item

        # 최상위 부모
        top_parent_id = self._get_top_parent_id(item_id)
        if top_parent_id is None:
            return None
        
        found_parent_id = self._find_id(self.root, top_parent_id)
        if found_parent_id is None:
            return None

        if found_parent_id == item_id:
            return global_item_manager.get(item_id)

        # 부모의 children_ids에서 탐색
        return self._search_in_children(top_parent_id, item_id)

    def _get_top_parent_id(self, item_id):
        """
        item_id를 가진 아이템의 '최상위 부모' ID를 찾는다.
        """
        current_id = item_id
        while True:
            cur_obj = global_item_manager.get(current_id)
            if cur_obj is None:
                return None
            if cur_obj.parent_id is None:
                # 최상위
                return current_id
            current_id = cur_obj.parent_id

    def _find_id(self, node, target_top_id):
        if node is None:
            return None
        
        comp = self._compare_item_ids(target_top_id, node.item_id)
        if comp < 0:
            return self._find_id(node.left, target_top_id)
        elif comp > 0:
            return self._find_id(node.right, target_top_id)
        else:
            return node.item_id

    def _search_in_children(self, parent_id, target_id):
        """
        parent_id 아이템의 자식(재귀)에서 target_id를 찾으면 그 id를 반환
        """
        parent_obj = global_item_manager.get(parent_id)
        if not parent_obj:
            return None

        for child_id in parent_obj.children_ids:
            if child_id == target_id:
                return global_item_manager.get(target_id)
            found_obj = self._search_in_children(child_id, target_id)
            if found_obj is not None:
                return found_obj
        return None

    #########################
    # 순회
    #########################
    def in_order_traversal(self):
        """
        AVL 트리를 중위 순회하여 (z>y>x 기준),
        아이템 객체들을 yield한다.
        """
        yield from self._in_order_traversal(self.root)

    def _in_order_traversal(self, node):
        if node:
            yield from self._in_order_traversal(node.left)
            yield global_item_manager.get(node.item_id)
            yield from self._in_order_traversal(node.right)

    def pre_order_traversal(self):
        yield from self._pre_order_traversal(self.root)

    def _pre_order_traversal(self, node):
        if node:
            yield global_item_manager.get(node.item_id)
            yield from self._pre_order_traversal(node.left)
            yield from self._pre_order_traversal(node.right)

    def post_order_traversal(self):
        yield from self._post_order_traversal(self.root)

    def _post_order_traversal(self, node):
        if node:
            yield from self._post_order_traversal(node.left)
            yield from self._post_order_traversal(node.right)
            yield global_item_manager.get(node.item_id)

    #########################
    # 그룹화 (예시)
    #########################
    def group_by_z(self):
        """
        z-좌표로 그룹화 (예: z+depth)
        """
        groups = {}
        for itm in self.in_order_traversal():
            if itm is None:
                continue
            w, h, d = itm.getDimension()
            x, y, z = itm.b_position
            top_z = z + d
            if top_z not in groups:
                groups[top_z] = []
            groups[top_z].append(itm)
        return groups


    #########################
    # 총 무게, 부피 계산
    #########################
    def get_total_weight(self):
        total_weight = 0
        for itm in self.in_order_traversal():
            if itm:
                total_weight += itm.getWeight()
        return total_weight

    def get_total_volume(self):
        total_volume = 0
        for itm in self.in_order_traversal():
            if itm:
                total_volume += itm.volume
        return total_volume

    def get_sorted_items(self, properties=None, reverse=False):
        """
        트리에 있는 모든 아이템을 지정된 프로퍼티(또는 여러 프로퍼티) 기준으로 정렬하여 리스트로 반환합니다.
        """
        items = list(self.in_order_traversal())
        items = [i for i in items if i is not None]
        
        if not properties:
            # 별도의 정렬 프로퍼티가 없으면, 그냥 in-order 결과 반환
            return items
        
        def sort_key(it):
            key_list = []
            for prop in properties:
                if not hasattr(it, prop):
                    raise ValueError(f"'{prop}' 속성이 Item에 없습니다.")
                val = getattr(it, prop)
                key_list.append(val)
            return tuple(key_list)
        
        return sorted(items, key=sort_key, reverse=reverse)


    def all_items_generator(self):
        """
        트리에 있는 모든 노드(Top-Level) + 그 자식(재귀)
        """
        for top_id in self._in_order_id_gen(self.root):
            top_item = global_item_manager.get(top_id)
            if top_item:
                yield top_item
                # 자식 재귀
                yield from self._children_id_generator(top_item)

    def _in_order_id_gen(self, node):
        if node:
            yield from self._in_order_id_gen(node.left)
            yield node.item_id
            yield from self._in_order_id_gen(node.right)

    def _children_id_generator(self, parent_item):
        """
        parent_item의 children_ids를 재귀적으로 순회
        """
        for cid in parent_item.children_ids:
            cobj = global_item_manager.get(cid)
            if cobj:
                yield cobj
                yield from self._children_id_generator(cobj)
                
    def print_all_items(self):
        """
        ItemTree에 등록된 모든 아이템(Top-Level + 자식 포함)을 출력한다.
        """
        print("📦 ItemTree에 포함된 모든 아이템:")
        for item in self.all_items_generator():
            if item:
                print(f" - ID: {item._id}, Name: {item.name}, Pos: {item.b_position}, "
                    f"Rotation: {item.rotation_quant}, Composite: {item.is_composite}, "
                    f"Children: {item.children_ids}")


#########################
# 트리 합치기 예시
#########################
def merge_trees_into_new(treeA, treeB):
    """
    두 개의 ItemTree (treeA, treeB)에 있는 모든 아이템들을
    새 ItemTree에 합쳐서 반환.
    """
    new_tree = ItemTree()
    for itmA in treeA.in_order_traversal():
        if itmA:
            new_tree.insert(itmA._id)
    for itmB in treeB.in_order_traversal():
        if itmB:
            new_tree.insert(itmB._id)
    return new_tree