# planning/itemManager.py
class ItemManager:
    def __init__(self):
        self._next_id = 0
        self._id_to_item = {}

    def register(self, item, *, allow_duplicate=False):
        """
        • item._id 가 없으면 새 id 부여  
        • 이미 같은 id 가 있는데 allow_duplicate=False -> Error  
        """
        # ① id 부여 / next_id 갱신
        if item._id is None:
            item._id = self._next_id
        self._next_id = max(self._next_id, item._id + 1)

        # ② 중복 처리
        if (not allow_duplicate) and (item._id in self._id_to_item):
            raise ValueError(f"Item id {item._id} already registered")
        self._id_to_item[item._id] = item
        return item._id

    def get(self, item_id):
        return self._id_to_item.get(item_id)
    
    def has_id(self, item_id):
        """
        주어진 item_id가 레지스트리에 존재하는지 bool로 반환.
        """
        return item_id in self._id_to_item
        
    def update(self, item_id, new_item):
        """
        기존 아이템 정보를 new_item으로 업데이트.

        Parameters:
        - item_or_id: 기존 아이템의 ID 또는 Item 객체
        - new_item: 새로운 Item 객체 (업데이트에 사용)

        Returns:
        - bool: 성공 여부
        """
        if item_id not in self._id_to_item:
            return False  # 해당 ID가 존재하지 않음
        
        item = self.get(item_id)

        new_item._id = item_id  # ID는 유지
        self._id_to_item[item_id] = new_item
        return True


# 전역 인스턴스
global_item_manager = ItemManager()

