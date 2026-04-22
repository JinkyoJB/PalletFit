from planning.bin import Bin
import numpy as np

class Zone(Bin):
    def __init__(self, x, y,z, width, height, depth, direction=0, unit='mm', max_weight=999999, name='zone', partno='0'):
        super().__init__(
            partno=partno,
            name=name,
            width=width,
            height=height,
            depth=depth,
            unit=unit,
            max_weight=max_weight,
        )
        '''
        self.x, self.y: bin 내부에 zone의 좌측 하단 꼭짓점 좌표
        '''
        self.x = x
        self.y = y
        self.z = z
        self.direction = direction
        self.options = {}
        self.adjacent_zones = []
        self._face_info = {}

    def __str__(self):
        base_str = super().__str__()  # Bin 클래스의 __str__ 호출
        additional_str = f"15. x : {self.x}\n16. y : {self.y} \n17. z : {self.z} \n18.leftover : {self.leftover}"
        return f"{base_str}\n{additional_str}"
    
    @property
    def leftover(self):
        return super().leftover

    def update_face_cache(self):
        """
        아이템의 8개 꼭짓점을 기반으로 6개 면(face)에 대한 정보를 계산하여 캐싱합니다.
        각 면 정보는 (plane, bounds) 형태로 저장됩니다.
          - plane: (a, b, c, d) (ax + by + cz + d = 0)
          - bounds: {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
        """
        vertices = self.getVertices()  # 8개 꼭짓점
        # rotation_quat=0 기준 인덱스 매핑 (다른 회전이 필요한 경우 추가 처리 필요)
        direction_map = {
            'bottom': [0, 1, 2, 3],
            'top':    [4, 5, 6, 7],
            'front':  [0, 1, 5, 4],
            'back':   [2, 3, 7, 6],
            'left':   [0, 3, 7, 4],
            'right':  [1, 2, 6, 5],
        }
        self._face_info.clear()
        for direction, indices in direction_map.items():
            face_vertices = [vertices[i] for i in indices]
            # 두 벡터 생성하여 법선 벡터 계산
            v1 = [face_vertices[1][i] - face_vertices[0][i] for i in range(3)]
            v2 = [face_vertices[2][i] - face_vertices[0][i] for i in range(3)]
            normal = np.cross(v1, v2)
            d = -sum(normal[i] * face_vertices[0][i] for i in range(3))
            plane = (normal[0], normal[1], normal[2], d)
            xs = [v[0] for v in face_vertices]
            ys = [v[1] for v in face_vertices]
            zs = [v[2] for v in face_vertices]
            bounds = {
                'x': (min(xs), max(xs)),
                'y': (min(ys), max(ys)),
                'z': (min(zs), max(zs))
            }
            self._face_info[direction] = (plane, bounds)
        return self._face_info

    def getFaceInfo(self, direction):
        """
        주어진 direction('front','back','left','right','top','bottom')에 해당하는 면의
        'plane'과 'bounds' 정보를 반환합니다.
        
        캐시된 면 정보(self._face_info)를 사용하며, 만약 캐시가 비어있다면 update_face_cache()를 호출합니다.
        
        Parameters
        ----------
        direction : str
            'front','back','left','right','top','bottom' 중 하나
        
        Returns
        -------
        plane : tuple(a, b, c, d)
            평면 방정식 계수 (ax + by + cz + d = 0)
        bounds : dict
            {'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}
        """
        valid_directions = ['bottom', 'top', 'front', 'back', 'left', 'right']
        if direction not in valid_directions:
            raise ValueError(f"Invalid direction: {direction}. Use one of {valid_directions}")
        if not self._face_info:
            self.update_face_cache()
        return self._face_info[direction]
    
    def store(self, item):
        super().store(item)
        self.leftover -= item.width * item.height * item.depth

