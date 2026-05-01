# planning/item.py
from planning.itemManager import global_item_manager
import numpy as np
from utils.item_utils import rotate_about_center


class RotationType:
    """
    RotationType을 Quaternion 형태로 정의
    """

    # 기본 회전 (Quaternion) 정의
    # 순서: [qx, qy, qz, qw]
    RT_WHD = [0.0, 0.0, 0.0, 1.0]            # 기본 (no rotation)
    RT_HWD = [0.0, 0.0, 0.7071068, 0.7071068]# Z축 90도 회전
    RT_HDW = [0.7071, 0.0, 0.0, 0.7071]      # X축 90도 회전
    RT_DHW = [0.0, 1.0, 0.0, 0.0]            # Y축 180도 회전
    RT_WDH = [0.0, 0.7071, 0.0, 0.7071]      # Y축 90도 회전
    RT_DWH = [0.0, -0.7071, 0.0, 0.7071]     # Y축 -90도 회전

    # 회전 모음
    All = [RT_WHD, RT_HWD, RT_HDW, RT_DHW, RT_WDH, RT_DWH]

    UnRotation = [RT_WHD]  # 회전 없는 상태만

    BasicRotation = [
        (RT_WHD, RT_HWD),   # 기본 ↔ Z축 90도
        (RT_HDW, RT_DHW),   # X축 90도 ↔ X축 180도
        (RT_WDH, RT_DWH),   # Y축 90도 ↔ Y축 -90도
    ]

    AxisAligned = [RT_WHD, RT_HWD, RT_HDW, RT_DHW, RT_WDH, RT_DWH]

    @classmethod
    def get_rotation_pair(cls, rotation_quat):
        """
        주어진 Quaternion에 대해 BasicRotation 안에서 페어 찾아서 반환
        """
        for q1, q2 in cls.BasicRotation:
            if np.allclose(rotation_quat, q1, atol=1e-2):
                return q2
            if np.allclose(rotation_quat, q2, atol=1e-2):
                return q1
        return None

    @classmethod
    def get_BasicRotation_index(cls, rotation_quat):
        """
        주어진 rotation_quat이 포함된 BasicRotation 리스트의 인덱스를 반환합니다.
        """
        for index, rotation_group in enumerate(cls.BasicRotation):
            for quat in rotation_group:
                if np.allclose(rotation_quat, quat, atol=1e-2):
                    return index
        return 0  # 어느 그룹에도 없으면 기본값 0 반환
    
    @classmethod
    def quat_to_index(cls, rotation_quat, tol: float = 1e-2) -> int:
        """
        rotation_quat(길이 4) 를 RotationType.All 순서에 따라 0..ROT_CARD-1 인덱스로 매핑.
        - np.allclose 로 근접한 항목을 찾고
        - 못 찾으면 0(기본 RT_WHD)로 폴백
        """
        if rotation_quat is None:
            return 0
        q = np.asarray(rotation_quat, dtype=np.float64)
        for idx, ref in enumerate(RotationType.All):
            if np.allclose(q, np.asarray(ref, dtype=np.float64), atol=tol):
                return idx
            
    @classmethod
    def index_to_quat(cls, index: int):
        """
        0..ROT_CARD-1 인덱스를 RotationType.All 순서에 따라 rotation_quat(길이 4)로 매핑.
        - 범위 밖 인덱스면 None 반환
        """
        if not isinstance(index, int) or index < 0 or index >= len(cls.All):
            return None
        return cls.All[index]
        

def is_axis_aligned(q, tol=1e-4):
    """RotationType.AxisAligned 중 하나와 거의 같으면 True"""
    return any(np.allclose(q, qa, atol=tol) for qa in RotationType.AxisAligned)


class Item:
    # ▶︎ 밀리미터·센티미터·미터 → meter 로 바꾸기 위한 계수
    _UNIT2M = {'mm': 1e-3, 'cm': 1e-2, 'm': 1.0}

    def __init__(self, *args, **kwargs):
        self._id = None  # 나중에 ItemTree에서 부여
        global_item_manager.register(self)

        # 필수 인자 목록
        required_fields = ['width', 'height', 'depth', 'unit']
        
        # 필수 인자 유효성 검사
        for field in required_fields:
            if kwargs.get(field) is None and (len(args) <= required_fields.index(field) or args[required_fields.index(field)] is None):
                raise ValueError(f"Item(): '{field}' is a required field.")
        
        # 속성 설정
        self.partno = kwargs.get('partno', args[0] if len(args) > 0 else None)
        self.name = kwargs.get('name', args[1] if len(args) > 1 else 'unknown')
        self.objshape = kwargs.get('objshape', args[2] if len(args) > 2 else 'cube')
        self.width = kwargs.get('width', args[3] if len(args) > 3 else None)
        self.height = kwargs.get('height', args[4] if len(args) > 4 else None)
        self.depth = kwargs.get('depth', args[5] if len(args) > 5 else None)
        self.rotation_quat = kwargs.get('rotation_quat', args[6] if len(args) > 6 else RotationType.RT_WHD)
        self.weight = kwargs.get('weight', args[7] if len(args) > 6 else None)
        self.priority = kwargs.get('priority', args[8] if len(args) > 7 else None)
        self.loadbear = kwargs.get('loadbear', args[9] if len(args) > 8 else None)
        self.updown = kwargs.get('updown', args[10] if len(args) > 9 else False if self.objshape != 'cube' else True)
        self.b_position = kwargs.get('b_position', args[11] if len(args) > 10 else [0, 0, 0])
        self.w_position = kwargs.get('w_position', args[12] if len(args) > 11 else None)
        self.unit = kwargs.get('unit', args[13] if len(args) > 12 else 'mm')
        self.options = kwargs.get('options', args[14] if len(args) > 13 else {})
        self._bottom_overlap_area = 0
        self.children_ids = []
        # 새로 추가: 부모 참조 (없으면 None)
        self.parent_id = None
        self.is_composite = False

        # 만약 parent가 있으면, 부모의 children에 이 아이템을 등록
        if self.parent_id is not None:
            self.parent_id.children_ids.append(self._id)


        # 유효성 검사 호출
        self.validate_inputs()

        # b_position을 np.float64로 변환
        if self.b_position is not None:
            if not isinstance(self.b_position, list) or len(self.b_position) != 3:
                raise ValueError("Item(): 'b_position' must be a list of length 3.")
            self.b_position = [np.float64(coord) for coord in self.b_position]

        # rotation_quat을 np.float64로 변환
        if self.rotation_quat is not None:
            if not isinstance(self.rotation_quat, list) or len(self.rotation_quat) != 4:
                raise ValueError("Item(): 'rotation_quat' must be a list of length 4.")
            self.rotation_quat = [np.float64(coord) for coord in self.rotation_quat]


        # 단위 변환
        if self.unit != 'mm':
            self.convert_unit('mm')


        # ────────────────────────────────────────────────
        # loadbear·weight 이 비어 있으면 경험식으로 추정
        # ────────────────────────────────────────────────
        scale = self._UNIT2M.get(self.unit, 1e-3)   # default: mm
        w_m = self.width  * scale          # [m]
        h_m = self.height * scale          # [m]
        d_m = self.depth  * scale          # [m]

        # (1) 허용 하중(loadbear) [kg]  =  상단 면적 × 계수
        PRESSURE_COEFF = 20000           # [kg / m²]  ← 필요시 조정
        area_m2        = w_m * h_m
        self.loadbear  = round(area_m2 * PRESSURE_COEFF)

        # (2) 무게(weight) [kg]  =  체적 × 밀도
        if self.weight is None or self.weight <= 0:
            DENSITY  = 200                 # [kg / m³]   ← 필요시 조정
            volume_m3 = w_m * h_m * d_m
            self.weight = round(volume_m3 * DENSITY, 2)   # 소수 2자리
        
        # 면 정보를 캐싱할 딕셔너리 (키: direction, 값: (plane, bounds))
        self._face_info = {}
        self.top_face_occupancy = self.width * self.height

        # getDimension() 결과 캐시 (rotation/dim 바뀌면 자동 무효화)
        self._dim_cache_key = None
        self._dim_cache_val = None

        # getVertices() / update_face_cache() 결과 캐시
        # 키: (b_position, rotation_quat, w, h, d) — 어느 하나 바뀌면 자동 miss
        self._verts_cache_key = None
        self._verts_cache_val = None      # tuple(tuple(...)) — immutable
        self._face_info_cache_key = None  # _face_info 자체가 캐시 저장소

    @property
    def ex(self):
        dimension = self.getDimension()
        return self.b_position[0] + dimension[0]

    @property
    def ey(self):
        dimension = self.getDimension()
        return self.b_position[1] + dimension[1]

    @property
    def ez(self):
        dimension = self.getDimension()
        return self.b_position[2] + dimension[2]


    def validate_inputs(self):
        '''Validate input types and values'''
        valid_objshapes = ['cube', 'cone', 'circle', 'cylinder']
        valid_units = ['mm', 'cm', 'm']

        if not isinstance(self.partno, (int,str, type(None))):
            raise TypeError("Item(): 'partno' must be an integer or str.")

        if not isinstance(self.name, str):
            raise TypeError("Item(): 'name' must be a string.")

        if self.objshape not in valid_objshapes:
            raise ValueError(f"Item(): 'objshape' must be one of {valid_objshapes}.")

        for dim in ['width', 'height', 'depth']:
            if not isinstance(getattr(self, dim), (int, float)):
                raise TypeError(f"Item(): '{dim}' must be an integer or float.")

        if self.priority is not None and not isinstance(self.priority, int):
            raise TypeError("Item(): 'priority' must be an integer or None.")

        if self.loadbear is not None and not isinstance(self.loadbear, (int, float)):
            raise TypeError("Item(): 'loadbear' must be an integer or float.")

        if not isinstance(self.updown, bool):
            raise TypeError("Item(): 'updown' must be a boolean.")

        for pos in ['b_position']:
            position = getattr(self, pos)
            if position is not None:
                if not isinstance(position, list) or len(position) != 3:
                    raise TypeError(f"Item(): '{pos}' must be a list of length 3.")
        
        if self.w_position is not None:
            if not isinstance(self.w_position, list) or len(self.w_position) != 7:
                raise TypeError("Item(): 'w_position' must be a list of length 7.")

        if self.unit not in valid_units:
            raise ValueError(f"Item(): 'unit' must be one of {valid_units}.")

        if not isinstance(self.options, dict):
            raise TypeError("Item(): 'options' must be a dictionary.")

    def __str__(self):
        attributes = [
            f"0. ID : '{self._id}'",
            f"1. partno : '{self.partno}'",
            f"2. name : '{self.name}'",
            f"3. objshape : '{self.objshape}'",
            f"4. width : '{self.width}'",
            f"5. height : '{self.height}'",
            f"6. depth : '{self.depth}'",
            f"7. rotation_quat : '{self.rotation_quat}'",
            f"8. weight : '{self.weight}'",
            f"9. priority : '{self.priority}'",
            f"10. loadbear : '{self.loadbear}'",
            f"11. updown : '{self.updown}'",
            f"12. b_position : '{self.b_position}'",
            f"13. w_position : '{self.w_position}'",
            f"14. unit : '{self.unit}'",
            f"15. options : '{self.options}'",
            f"16. children_ids : '{self.children_ids}'",
            f"17. is_composite : '{self.is_composite}'",
            f"18. face_info : '{self._face_info}'",
            f"19. top_face_occupancy : '{self.top_face_occupancy}'",
            '----------------------------------------------'
        ]
        return "\n".join(attributes)

    def __repr__(self):
        return self.__class__.__name__

    # ────────────────────────────────────────────────
    # 범용 단위 변환 메서드
    # ────────────────────────────────────────────────
    def convert_unit(self, target_unit: str):
        """
        인스턴스의 가로·세로·높이를 `target_unit`(mm, cm, m 중 하나)로 변환.
        self.unit 도 함께 갱신한다.

        예)
            item.convert_unit('mm')  # 현 단위 → mm 로 통일
        """
        if target_unit not in self._UNIT2M:
            raise ValueError(f"convert_unit(): 지원하지 않는 단위 '{target_unit}'입니다.")

        # 현재·목표 단위를 모두 미터 환산 계수로 치환
        curr2m   = self._UNIT2M[self.unit]
        target2m = self._UNIT2M[target_unit]

        # (현재 → m) → (m → 목표) == (curr2m / target2m) 배 스케일링
        scale = curr2m / target2m

        self.width  *= scale
        self.height *= scale
        self.depth  *= scale
        self.unit    = target_unit


    @property
    def volume(self):
        """
        아이템의 부피를 반환 (소수점 n자리로 세팅).
        사용 예) 
            my_item = Item(width=3, height=4, depth=5, weight=1, unit='mm')
            print(my_item.volume)  # 60.000
        """
        return self.width * self.height * self.depth

    def getWeight(self):
        return self.weight
    
    def getBottomOverlap(self):
        """
        아이템의 바닥 면적을 반환합니다.
        """
        return self._bottom_overlap_area

    def getDimension(self):
        if self.name == 'gripper':
            return [self.width, self.height, self.depth]

        q = self.rotation_quat
        # 캐시 키: width/height/depth/rotation 4개. 어느 하나 바뀌면 자동 miss.
        # rotation_quat이 list여도 tuple로 변환해 hashable + 값 비교.
        cache_key = (self.width, self.height, self.depth,
                     q[0], q[1], q[2], q[3])
        if self._dim_cache_key == cache_key and self._dim_cache_val is not None:
            return list(self._dim_cache_val)  # 안전을 위해 사본 반환

        if is_axis_aligned(q):
            w, h, d = self.width, self.height, self.depth

            # Z축 ±90°: w↔h
            if np.allclose(q, RotationType.RT_HWD, atol=1e-2) or np.allclose(q, RotationType.RT_DWH, atol=1e-2):
                w, h = h, w

            # X축 ±90°: h↔d
            elif np.allclose(q, RotationType.RT_HDW, atol=1e-2) or np.allclose(q, RotationType.RT_DHW, atol=1e-2):
                h, d = d, h

            # Y축 ±90°: w↔d
            elif np.allclose(q, RotationType.RT_WDH, atol=1e-2) or np.allclose(q, RotationType.RT_DWH, atol=1e-2):
                w, d = d, w

            # Z축 180° 등 축정렬 나머지는 (w,h,d) 그대로
            result = [w, h, d]
        else:
            # ── 일반 회전 경로(축 비정렬): 안전한 AABB로 fallback ──
            v = np.asarray(self.getVertices())
            dims = (v.max(axis=0) - v.min(axis=0)).tolist()
            result = [round(x, 6) for x in dims]

        self._dim_cache_key = cache_key
        self._dim_cache_val = tuple(result)  # 내부엔 immutable로 보관
        return result


    def getResult(self):
        if self.b_position is None:
            raise ValueError("getResult(): b_position is not set.")
        if self.rotation_quat is None:
            raise ValueError("getResult(): rotation_quat is not set.")
        
        return self.b_position + self.rotation_quat


    def getVertices(self):
        """
        회전 후에도 다음 순서대로 꼭짓점 반환:
        아래면: 0(FL), 1(FR), 2(BR), 3(BL)
        윗면:   4(FL), 5(FR), 6(BR), 7(BL)
        윗면
        7️⃣6️⃣
        4️⃣5️⃣

        밑면
         3️⃣2️⃣
        .0️⃣1️⃣
        원점
        """
        # 캐시 hit: b_position/rotation/dim 모두 동일하면 재계산 스킵
        bp = self.b_position
        rq = self.rotation_quat
        cache_key = (bp[0], bp[1], bp[2],
                     rq[0], rq[1], rq[2], rq[3],
                     self.width, self.height, self.depth)
        if self._verts_cache_key == cache_key and self._verts_cache_val is not None:
            return [list(v) for v in self._verts_cache_val]  # 사본(caller mutation 안전)

        w, h, d = map(float, (self.width, self.height, self.depth))
        local = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [w, h, 0],
            [0, h, 0],
            [0, 0, d],
            [w, 0, d],
            [w, h, d],
            [0, h, d],
        ])
        rotated = rotate_about_center(local, self.rotation_quat)
        min_corner = rotated.min(axis=0)
        shift = np.asarray(self.b_position) - min_corner
        pts = (rotated + shift)

        # ── z 두 레벨을 허용오차로 분리 ──
        z = pts[:, 2]
        zmin, zmax = float(z.min()), float(z.max())
        eps = max(1e-3, 1e-6 * max(w, h, d))  # mm 단위 기준 여유

        bottom_mask = np.isclose(z, zmin, atol=eps)
        top_mask    = np.isclose(z, zmax, atol=eps)

        bottom = pts[bottom_mask]
        top    = pts[top_mask]

        # 🔒 안전 가드: 4점이 아니면 AABB 폴백
        if len(bottom) != 4 or len(top) != 4:
            # AABB로 재구성: (b_position)를 FLB로 두고 Axis-Aligned 박스 작성
            W, H, D = self.getDimension()
            x0, y0, z0 = map(float, self.b_position)
            aabb = np.array([
                [x0,     y0,     z0    ],
                [x0+W,   y0,     z0    ],
                [x0+W,   y0+H,   z0    ],
                [x0,     y0+H,   z0    ],
                [x0,     y0,     z0+D  ],
                [x0+W,   y0,     z0+D  ],
                [x0+W,   y0+H,   z0+D  ],
                [x0,     y0+H,   z0+D  ],
            ])
            # 아래/윗면 분리 및 정렬 동일 처리
            zb = aabb[:,2]; zmin, zmax = float(zb.min()), float(zb.max())
            bottom = aabb[np.isclose(zb, zmin, atol=eps)]
            top    = aabb[np.isclose(zb, zmax, atol=eps)]

        # y(앞→뒤), x(좌→우) 기준 정렬
        key2 = lambda p: (p[1], p[0])
        bottom_sorted = bottom[np.lexsort((bottom[:,0], bottom[:,1]))]
        top_sorted    = top[np.lexsort((top[:,0],    top[:,1]))]

        # ✅ 순서 재배열: FL, FR, BR, BL
        def reorder(quad):
            quad = quad.tolist()
            if len(quad) != 4:
                # 더블 가드: 폴백
                return quad[:4] + [[quad[0][0], quad[0][1], quad[0][2]]] * (4 - len(quad))
            return [quad[0], quad[1], quad[3], quad[2]]

        round3 = lambda pt: [round(float(pt[0]), 3), round(float(pt[1]), 3), round(float(pt[2]), 3)]
        verts = [round3(v) for v in (reorder(bottom_sorted) + reorder(top_sorted))]

        # 캐시 저장 (immutable tuple로)
        self._verts_cache_key = cache_key
        self._verts_cache_val = tuple(tuple(v) for v in verts)
        return verts


    def update_face_cache(self):
        """
        회전 여부와 관계없이 AABB 기준 6 면을
        front/back/left/right/top/bottom 라벨로 고정해 캐싱.
        b_position/rotation/dim이 그대로면 재계산 스킵.
        """
        bp = self.b_position
        rq = self.rotation_quat
        cache_key = (bp[0], bp[1], bp[2],
                     rq[0], rq[1], rq[2], rq[3],
                     self.width, self.height, self.depth)
        if self._face_info_cache_key == cache_key and self._face_info:
            return  # 캐시 hit — _face_info 그대로 사용

        v = np.asarray(self.getVertices())
        xmin, ymin, zmin = v.min(axis=0)
        xmax, ymax, zmax = v.max(axis=0)

        normals = {
            'left'  : (-1.0,  0.0,  0.0),
            'right' : ( 1.0,  0.0,  0.0),
            'front' : ( 0.0, -1.0,  0.0),
            'back'  : ( 0.0,  1.0,  0.0),
            'bottom': ( 0.0,  0.0, -1.0),
            'top'   : ( 0.0,  0.0,  1.0),
        }
        planes_d = {
            'left'  :  xmin,
            'right' : -xmax,
            'front' :  ymin,
            'back'  : -ymax,
            'bottom':  zmin,
            'top'   : -zmax,
        }

        rng = {
            'x': (float(xmin), float(xmax)),
            'y': (float(ymin), float(ymax)),
            'z': (float(zmin), float(zmax)),
        }

        self._face_info.clear()
        for name, n in normals.items():
            a, b, c = n
            d = float(planes_d[name])          # ax+by+cz+d=0 꼴
            # bounds : 평면과 수직인 축은 (c,c) 로 고정
            if name in ('left', 'right'):
                bounds = {'x': (rng['x'][0] if a<0 else rng['x'][1],)*2,
                        'y': rng['y'], 'z': rng['z']}
            elif name in ('front', 'back'):
                bounds = {'x': rng['x'],
                        'y': (rng['y'][0] if b<0 else rng['y'][1],)*2,
                        'z': rng['z']}
            else:  # bottom / top
                bounds = {'x': rng['x'], 'y': rng['y'],
                        'z': (rng['z'][0] if c<0 else rng['z'][1],)*2}
            self._face_info[name] = ((a, b, c, d), bounds)

        # 캐시 키 갱신
        self._face_info_cache_key = cache_key

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
        valid_directions = ['left', 'right', 'front', 'back', 'bottom', 'top']
        if direction not in valid_directions:
            raise ValueError(f"Invalid direction: {direction}. Use one of {valid_directions}")
        if not self._face_info:
            self.update_face_cache()
        return self._face_info[direction]
    
    def update_child_positions_z(self):
        """
        parent_item의 b_position이 확정되어 있다고 가정.
        parent_item.children_ids (id 리스트)를 z축 방향으로 스택하면서,
        각 자식의 b_position을 '부모 좌표 + 누적 깊이'로 설정.
        """

        if not self.children_ids:
            return  # 자식 없으면 종료

        px, py, pz = self.b_position

        current_z = pz

        # ✅ registry 통해 실제 자식 객체 가져오기
        children = [global_item_manager.get(child_id) for child_id in self.children_ids]

        # ✅ z축 기준 90도 회전 quaternion 후보들
        for child in children:
            _, _, cD = child.getDimension()
            # b_position 설정
            child.b_position = [px, py, current_z]
            child.rotation_quat = self.rotation_quat
            current_z += cD

            # 자식들의 item._safety_factor는 부모 값으로 설정
            child._bottom_overlap_area = self._bottom_overlap_area
            # 자식들의 item.priority는 부모 값으로 설정
            child.priority = self.priority

            # 자식들의 면정보 캐시 업데이트
            child.update_face_cache()

            global_item_manager.update(child._id, child)

            # 재귀 호출
            child.update_child_positions_z()

    # ────────────────────────── Pickle hooks ──────────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        # ①  캐시를 빼고 싶으면 주석 유지, 아니라면 제거
        # state.pop('_face_info', None)
        return state

    def __setstate__(self, state):
        # ②  원래 속성 복원
        self.__dict__.update(state)

        # ③  _face_info 가 빠져 있으면 빈 dict 로
        if '_face_info' not in self.__dict__:
            self._face_info = {}

        # ③-b 캐시 필드 backward-compat (옛 pickle엔 없음)
        for k in ('_dim_cache_key', '_dim_cache_val',
                  '_verts_cache_key', '_verts_cache_val',
                  '_face_info_cache_key'):
            if k not in self.__dict__:
                self.__dict__[k] = None

        # ④  워커 프로세스의 global_item_manager 에 재등록
        from planning.itemManager import global_item_manager
        global_item_manager._id_to_item[self._id] = self
        if self._id >= global_item_manager._next_id:
            global_item_manager._next_id = self._id + 1
