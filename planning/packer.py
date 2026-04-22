# planning/packer.py
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
_render_pool = ThreadPoolExecutor(max_workers=2)

from motion.network_element import LogicalState
from planning.bin import Bin
from planning.item import Item, RotationType
from planning.BinSpecsDict import BIN_SPECS as _BIN_SPECS

from utils.item_utils import Item_b2w_position, shift_bin_w2b_position, shift_bin_coor2world
from utils.get_value import get_direction_overlap, flat_surface_weight, get_distance
from utils.util_functions import Log, load_offline_data, str_norm 
from utils.constants import EPS
from utils.util_functions import dump_items_to_json
from utils.overlap import aabb_bounds_xyz, compute_overlap_area

from dataclasses import asdict, dataclass
from planning.itemManager import global_item_manager
import json, random, importlib, time, copy
import numpy as np

from scipy.spatial.transform import Rotation as R

import dsrpy as dsr
from detection.N_detection import N_detection


# ====== [B] 레이블 상수 (문자열) ======
PROBLEMS = {"OFFLINE", "ONLINE_GIVEN_TYPE", "ONLINE", "EXHIBITION"}
MODES    = {"PALLETIZING", "BINPACKING"}

# ====== [C] 모델 레지스트리(간단 메타) ======
MODEL_REGISTRY = {
    "FFD":       {"module": "planning.heuristics.FFD",                       "cls": "FFD",                  "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "FFD_Z":     {"module": "planning.heuristics.FFD_Z",                     "cls": "FFD_Z",                "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "FFD_M":     {"module": "planning.heuristics.FFD_M",                     "cls": "FFD_M",                "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "SingleFit": {"module": "planning.heuristics.SingleFit.SingleFit",                 "cls": "SingleFit",            "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "ZoneFit":   {"module": "planning.heuristics.ZoneFit.ZoneFit",           "cls": "ZoneFit",              "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "BestFit":   {"module": "planning.heuristics.BestFit.BestFit",           "cls": "BestFit",              "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "OutlineFit":{"module": "planning.heuristics.OutlineFit.OutlineFit",     "cls": "OutlineFit",           "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "SafeFit":   {"module": "planning.heuristics.SafeFit",                   "cls": "SafeFit",              "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "PalletFit": {"module": "planning.heuristics.PalletFit.PalletFit_v2",              "cls": "PalletFit",            "init": {}, "kind":"heuristic", "supports_online_stream": True},
    "PalletFit_RL": {
        "module": "planning.RL.PalletFit_RL.rl_adapter",
        "cls": "PalletFitRLAdapter",
        "init": {
            "weight_file_dir": "planning/RL/PalletFit_RL/logs/MaskablePPO_AutoResume_done/ppo_ckpt_12251136_steps_29_29.zip",
            "selectable_len" : 3,
            "cands_option"  : 'EDP',  # 'EDP', 'CP', 'EP', 'EMS'
        },
        "kind": "rl",
        "supports_online_stream": True,
    },
    "DeepPack3D": {
        "module": "planning.adapters.deeppack3d_adapter",
        "cls": "DeepPack3DAdapter",
        "init": {
            "method": "rl",           # {"rl","bl","baf","bssf","blsf"} 중 선택
            "lookahead": 5,           # DeepPack3D의 탐색 길이
            "data_mode": "file",      # 우리 아이템을 임시파일로 넘김
            "unit_scale": 1.0,        # mm기준이면 1.0, m→mm 변환 등 필요시 조정
            "verbose": 0
        },
        "kind": "heuristic",
        "supports_online_stream": True,
    },
"DRL": {
        "module": "planning.adapters.drl_adapter",
        "cls": "DRLAdapter",
        "init": {
            "drl_path": "planning/adapters/external/DRL_Repo",
            "model_path": 'planning/adapters/external/DRL_Repo/pretrained_models/default_rs.pt', 
            "device": "cuda",
            "resolution": 90,
        },
        "kind": "rl",
        "supports_online_stream": True,
    },
}

# ====== [D] 호환성 매트릭스: (problem, mode) -> 허용 모델 set ======
COMPAT = {
    ("OFFLINE",           "BINPACKING"):  {"FFD","FFD_Z","FFD_M","SingleFit","ZoneFit","BestFit","OutlineFit","SafeFit","PalletFit","DQN", "SafeFit"},
    ("OFFLINE",           "PALLETIZING"): {"FFD","FFD_Z","FFD_M","SingleFit","ZoneFit","BestFit","OutlineFit","SafeFit","PalletFit","DQN", "SafeFit"},
    ("ONLINE_GIVEN_TYPE", "PALLETIZING"): {"BestFit","FFD_Z", "PalletFit","PalletFit_RL", "OutlineFit"},
    ("ONLINE",          "PALLETIZING"): {"FFD","FFD_Z","FFD_M","PalletFit_RL","SingleFit","ZoneFit","BestFit","OutlineFit","SafeFit","PalletFit","DQN", "SafeFit"},
    ("EXHIBITION",        "PALLETIZING"): {"OutlineFit"},
}


class Packer(LogicalState):
    '''
    planning의 주요 클래스
    1. bin, item을 관리
    2. packing 알고리즘 관리
    3. packing 실행
    '''
    BIN_SPECS = _BIN_SPECS

    def __init__(self, **kwargs):
        super().__init__()

        self.gripper = Item(
            partno=-1,
            name='gripper',
            objshape='cube',
            width = 0.1, 
            height = 0.1,
            depth = 0.1,   #그리퍼에 달린 아이템은 들어갈 수 있어야함.
            level=1,
            updown=False,
            unit='mm',
        )

        self.ui = kwargs.get('ui', None)    # GUI object
        # ------------ 아이템 관련 ------------
        self.problem = kwargs.get('problem', 'online')
        self.packing_mode = kwargs.get('packing_mode', 'palletizing')  # default: packingModel에 따라 packing, custom: custom packingModel 사용
        self.model = kwargs.get('model', "PalletFit_RL")                          # online_given_type: 주어진 item type_list 중에서 random하게 item을 생성하여 packing, offline: 주어진 item list를 packing
        self.offline_item_path = kwargs.get('offline_item_path', None)
        self.online_item_type_path = kwargs.get('online_item_type_path', None)  # GUI에서 사용안함. online_given_type에서 사용
        self.direct_item_list = kwargs.get('direct_item_list', None)            # GUI에서 사용안함.
        self.cands_option = kwargs.get('cands_option', 'EDP')  # 'EDP', 'CP', 'EP', 'EMS' 중 하나

        # 정규화된 키 보관(원하면 속성으로 노출)
        self._P = str_norm(self.problem)
        self._M = str_norm(self.packing_mode)

        # ✅ 조기 검증
        self._validate_config(self._P, self._M, self.model)

        self.unfit_stop_setting = kwargs.get('unfit_stop_setting', True)    # True: unfit아이템이 생기면 packing 중단, False: unfit아이템은 list에 추가하고 다음 아이템으로 packing 계속
        self.order_setting = kwargs.get('order_setting', False)  # True: packing 전에 아이템을 우선순위에 따라 정렬, False: packing 전에 아이템을 정렬하지 않음
        self.rotation_type = kwargs.get('rotation_type', RotationType.BasicRotation)  # packingModel에 필요한 회전 타입

        # ────────── logger 설정 ──────────
        self.log = (kwargs.get("logger", Log("planning/logs"))).write
        self.reset()  # 내부에서 reset() 호출

        # ------------ bin 관련 ------------
        self.bin_path = kwargs.get('bin_path', None)
        if self.bin_path:
            self.change_bin_path(self.bin_path)

        # ------------ packingModel 관련 ------------
        self.base_model1 = self.base_model("ZoneFit")
        self.base_model2 = self.base_model("FFD")
        self.change_model()

        # ------------ robot 관련 ------------
        self.robot = dsr.Robot()
        self.robot_ip = "192.168.1.30"
        self.robot_plag = kwargs.get('robot_plag', False)
        self.robot_is_busy = False  # 로봇이 물리적으로 움직이는 중인지 확인하는 플래그

        if self.robot_plag:
            self.robot_on()

    # =========== 로봇 관련 ============
    def robot_on(self):
        self.robot.open_connection(self.robot_ip)
        self.robot.connect_rt(self.robot_ip)
        self.robot.servo_on()
        
        # 1. 변환: 160mm -> 0.16m
        x, y, z = 0.0, 0.0, 0.16
        
        # 2. 회전: Euler XYZ -> Matrix (Y축 180도 회전)
        r = R.from_euler('xyz', [0, 180, 0], degrees=True)
        rot_mat = r.as_matrix()
        
        # 3. 4x4 행렬 생성
        tcp_tmat = np.eye(4)
        tcp_tmat[:3, :3] = rot_mat
        tcp_tmat[:3, 3] = [x, y, z]
        
        # 4. 적용
        self.robot.set_tcp_tmat(tcp_tmat)
        print(f"✅ TCP 설정 완료: Z={z}m, RotY=180deg")

    def robot_off(self):
        self.robot.servo_off()
        self.robot.disconnect_rt()
        self.robot.close_connection()

    # =========== packingModel 관련 ============
    def _validate_config(self, problem: str, mode: str, model: str):
        P = str_norm(problem); M = str_norm(mode)
        if P not in PROBLEMS:
            raise ValueError(f"Unknown problem '{problem}'. Allowed: {sorted(PROBLEMS)}")
        if M not in MODES:
            raise ValueError(f"Unknown packing_mode '{mode}'. Allowed: {sorted(MODES)}")
        key = (P, M)
        if key not in COMPAT:
            raise ValueError(f"No rule for ({P}, {M}).")
        allowed = COMPAT[key]
        if model not in allowed:
            raise ValueError(f"Model '{model}' not allowed for ({P}, {M}). Allowed: {sorted(allowed)}")

        # 온라인 문제인데 모델이 스트림 미지원이면 추가 체크
        if P in {"ONLINE","ONLINE_GIVEN_TYPE"}:
            meta = MODEL_REGISTRY.get(model, {})
            if not meta.get("supports_online_stream", False):
                raise ValueError(f"Model '{model}' does not support online stream problems.")
            
    def base_model(self, base_model_name):
        # meta = MODEL_REGISTRY["PalletFit"]
        meta = MODEL_REGISTRY[base_model_name]
        module = importlib.import_module(meta["module"])
        ModelClass = getattr(module, meta["cls"])
        init_args = {
            'unfit_stop_setting': self.unfit_stop_setting,
            'rotation_type':      self.rotation_type,
            **meta.get("init", {})
        }
        return ModelClass(**init_args)

    def change_model(self):
        meta = MODEL_REGISTRY[self.model]
        module = importlib.import_module(meta["module"])
        ModelClass = getattr(module, meta["cls"])
        init_args = {
            'unfit_stop_setting': self.unfit_stop_setting,
            'rotation_type':      self.rotation_type,
            **meta.get("init", {})
        }
        # ★ RL 어댑터일 경우 packer 자신도 전달
        if meta.get("kind") == "rl":
            init_args["packer"] = self
            # PalletFit_RL의 경우 cands_option을 packer의 값으로 덮어씀
            if self.model == "PalletFit_RL":
                init_args["cands_option"] = self.cands_option
        self.packingModel = ModelClass(**init_args)
        
    def reset(self):
        self.bins = []  # bin의 상태 정보 클래스 Bin을 담는 리스트
        self.current_bin_idx = 0
        self.max_items = 100
        self.think_iter = 0
        self.iter = 0

        self.items_list = [] # item의 상태 정보 클래스 Item을 담는 리스트
        self.unfit_items = []   # packing 중에 unfit이 발생하면 저장하는 리스트
        self.item_types = []    # item의 종류
        self.bk_bin = None
        self.target_item = None

        self.last_posori = None  # 마지막으로 포장한 아이템의 위치 및 자세 정보

        if self.problem == 'offline':
            self._load_offline_data()
            
        elif self.problem == 'online_given_type':
            self._load_online_item_type_data()

        elif 'online' in self.problem:
            pass

        elif self.problem == 'exhibition':
            # exhibition용 bin 생성
            self.build_bin('A')
            self.build_bin('B')
            self.build_bin('A')
            self.build_bin('B')

        elif self.direct_item_list:
            self.items_list = self.direct_item_list
            
        else:
            raise ValueError("Packer(): 'problem' must be one of ['online', 'offline', 'online_given_type','exhibition'].")

        # ------------------ robot 관련 변수(update_robot_state 함수 return값으로 업데이트)--------------------
        self.conveyor_items = []    # conveyor에 있는 item들의 Item1 Class List
        self.can_packing = True # packing_complete 관련 변수

        # ============ robot Behavior 관련 ============
        # self.pre_position = None    # push&place 저장용 변수
        self.pre_position = None    # push&place 저장용 변수

        
    def change_bin_path(self, bin_path):
        self.bin_path = bin_path
        if hasattr(self, 'bins'):
            del self.bins
        self.bins = []  
        self._load_bin_data(self.bin_path)

    def addBin(self, bin):
        self.bins.append(bin) 

    def orderBins(self):
        '''박스를 용량에 따라 정렬'''
        self.bins = sorted(self.bins, key=lambda x: x.volume, reverse=True)


    @property
    def current_bin(self):
        if len(self.bins) == 0:
            return None
        if self.current_bin_idx >= len(self.bins):
            return True
        return self.bins[self.current_bin_idx]
    
    def set_init_bin(self,bin_idx, item_list):
        '''
        self.bins에 있던 bin중 기존에 있던 아이템 정보를 저장하는 함수
        '''
        self.log('set_init_bin 함수가 실행됐습니다.')
        bin = self.bins[bin_idx]
        bin.reset()
        for item in item_list:
            bin.store(item)
            item.options['is_fixed'] = True

    def change_target_bin(self):
        '''
        target_bin을 바꿉니다.
        bin_items가 max_items보다 크거나 같으면, 다음 bin으로 바꿉니다.
        '''
        
        # self.packer.bins의 길이를 확인
        bin_count = len(self.bins)
        # self.current_bin_idx를 1 증가시킵니다.
        if self.current_bin_idx < bin_count - 1:
            self.current_bin_idx += 1
        else:
            self.current_bin_idx = 0
        
        self.sync_targetBin_state()

        self.can_packing = True

    def _create_bin_from_spec(self, spec: dict):
        """spec dict -> Bin 생성 & 등록."""
        spec = spec.copy()
        spec["partno"] = f"BIN_{len(self.bins) + 1}"
        self.addBin(Bin(**spec))
    def _create_bin_from_spec(self, spec: dict):
        """spec dict -> Bin 생성 & 등록."""
        spec = spec.copy()
        spec["partno"] = f"BIN_{len(self.bins) + 1}"
        self.addBin(Bin(**spec))

    def build_bin(self, key: str = "default2"):
        """
        key 하나만 받아 지정된 Bin 을 만든다.
        예) build_bin('A'), build_bin('B')
        """
        try:
            self._create_bin_from_spec(self.BIN_SPECS[key])
        except KeyError:
            print(f"[WARN] '{key}' spec이 없습니다: {list(self.BIN_SPECS)}")
                
    def _load_bin_data(self, bin_path=None):
        '''bin_path로부터 bins를 읽어온 뒤, Bin 객체로 변환하여 저장'''
        try:
            with open(bin_path, 'r') as file:
                bin_list = json.load(file)
                # self.log("Loaded bins:", bin_list)
            
            for bin in bin_list:
                self.addBin(Bin(**bin))
            
            return bin_list
        
        except FileNotFoundError:
            self.log(f"Error: File {self.bin_path} not found.")
    
    # =========== packing state 관련 =============
    @property
    def place_w_position(self):
        def find_bin_index_by_substring(substring):
            """
            bins 내의 bin 중 bin.name 에 `substring`이 포함되어 있으면
            그 bin의 인덱스를 반환합니다.
            없으면 -1을 반환합니다.
            """
            for i, b in enumerate(self.bins):
                if substring in b.name:
                    return i
            return -1
        if self.target_item is None:
            return None
        
        bp = self.target_item.b_position
        if np.allclose(bp, [-1,-1,-1], atol=1e-2):
            print('place_w_position) target b_position', bp)
            return None
        
        if self.problem == 'exhibition':
            if 'A' in self.current_bin.name:
                a_index = find_bin_index_by_substring('A')
                bin = self.bins[a_index]
                return Item_b2w_position(self.target_item, bin)
            elif 'B' in self.current_bin.name:
                b_index = find_bin_index_by_substring('B')
                bin = self.bins[b_index]
                return Item_b2w_position(self.target_item, bin)
        return Item_b2w_position(self.target_item, self.current_bin)
    
    @property
    def target_b_position(self):
        if self.target_item is None:
            return None
        
        if self.target_item.b_position == [-1,-1,-1]:
            return None
        return self.target_item.b_position
    
    # =========== conveyor 관련 =============

    def update_conveyor_items(self, target_spelling=None):
        '''
        ROS2 Conveyor server에 요청하여 self.conveyor_items를 업데이트합니다.
        '''
        self.conveyor_items.clear()

        # self.problem에 "online"이 포함되어있으면
        if 'online' in self.problem:
            responsed_items = N_detection()
            if responsed_items is not None:
                item_list = self.trans_itemData2Item(self.current_bin, responsed_items)
                self.conveyor_items = item_list

            else:
                self.conveyor_items = []
            self.log(f'update_conveyor_items : {len(self.conveyor_items)}개 아이템 정보 수신')


        elif self.problem == 'offline':
            self.conveyor_items = self.items_list

        return self.conveyor_items
    
    # =========== item 관련 =============
    def trans_itemData2Item(self,bin, items_data):
        '''
        items_data를 Item class로 변환하여 반환
        '''
        item_list = []

        for i, item_data in enumerate(items_data):
            whd = np.array(item_data.whd.data)  # 받은 크기값 (width, height, depth)
            whd_rounded = np.round(whd, 3)  # 소수점 2자리 반올림

            unit = item_data.unit
            # ✅ 단위를 mm로 변환
            if unit == 'm':
                whd_rounded *= 1000
            elif unit == 'cm':
                whd_rounded *= 10

            # print('item w_position : ', item_data.posori)
            item = Item(
                partno = i, 
                name = item_data.name,
                objshape = 'cube',
                width = whd_rounded[0],
                height = whd_rounded[1],
                depth = whd_rounded[2],
                updown = False,
                w_position = list(item_data.posori),
                unit = 'mm',
                rotation_quat = RotationType.RT_WHD,
                priority = 1,
            )
            item.b_position = shift_bin_w2b_position(bin, item)
            item.options['color'] = 'blue'

            item_list.append(item)
        return item_list


    def _load_offline_data(self):
        """
        offline_item_path로부터 items를 읽어온 뒤, Item 객체로 변환하여 self.items_list에 저장
        """
        item_dicts = load_offline_data(self.offline_item_path)  # ➋ 공통 로딩
        if item_dicts is None:
            self.log(f"Error: File {self.offline_item_path} not found.")
            return None

        for item_info in item_dicts:
            # load_offline_data에서 이미 partno, b_position이 설정되어 있음
            self.addItem(Item(**item_info))  # ➌ Item으로 변환하여 추가

        return item_dicts
    
    def _load_online_item_type_data(self):
        """
        online_item_type_list를 읽어온 뒤, 
        해당 타입들의 속성을 기반으로 새로운 Item 인스턴스를 생성하여 self.items_list에 추가.
        """
        try:
            # 1. 파일 읽기
            with open(self.online_item_type_path, 'r') as file:
                if self.online_item_type_path.endswith('.json'):
                    item_type_list = json.load(file)
                elif self.online_item_type_path.endswith('.txt'):
                    item_type_list = file.read().splitlines()
                elif self.online_item_type_path.endswith('.csv'):
                    item_type_list = file.read().split(',')
            
            # 2. 템플릿(Type) 리스트 생성 (이들은 생성용 '거푸집' 역할만 함)
            # 여기서는 Item 객체로 만들지 않고 dict 상태로 유지해도 되지만, 
            # 기존 로직과의 호환성을 위해 Item 객체로 만들어둡니다.
            # (단, 이들은 items_list에 들어가지 않으므로 ID가 낭비되긴 하지만 큰 문제는 없습니다)
            for item_data in item_type_list:
                self.item_types.append(Item(**item_data))

            # 3. 랜덤 생성 설정
            random_item_num = self.max_items
            
            # 각 타입별 생성 확률 랜덤 배정
            item_distribution = [random.randint(1, 10) for _ in range(len(self.item_types))]
            total_weight = sum(item_distribution)
            probabilities = [value / total_weight for value in item_distribution]

            # 4. 랜덤 아이템 생성 및 리스트 추가
            for _ in range(random_item_num):
                chosen_index = random.choices(range(len(self.item_types)), probabilities)[0]
                template = self.item_types[chosen_index]
                
                # ★ [핵심 변경] deepcopy 대신 '속성 기반 새 인스턴스 생성'
                # Item(...)을 호출하면 __init__ -> global_item_manager.register()가 실행되어
                # 자동으로 겹치지 않는 고유 ID가 부여됩니다.
                new_item = Item(
                    partno = template.partno,  # 타입 정보(부품 번호)는 유지
                    name = template.name,
                    objshape = template.objshape,
                    width = template.width,
                    height = template.height,
                    depth = template.depth,
                    weight = template.weight,
                    loadbear = template.loadbear,
                    unit = template.unit,
                    # 회전이나 색상은 필요하면 초기화, 아니면 템플릿 따름
                    rotation_quat = list(template.rotation_quat), 
                    options = copy.deepcopy(template.options) # mutable 객체는 복사 필요
                )
                
                # 생성된 새 아이템을 리스트에 추가
                self.items_list.append(new_item)

            # self.log(f"Generated {len(self.items_list)} random items from {len(self.item_types)} types.")

        except FileNotFoundError:
            self.log(f"Error: File {self.online_item_type_path} not found.")
            return None
       

    def addItem(self, item):
        """
        item.loadbear 이 없으면 자동 추정,
        item.weight 가 없거나 0 이하이면 자동 추정 후 self.items_list 에 등록
        """
        self.items_list.append(item)

    def orderItems(self):
        '''아이템을 우선순위에 따라 정렬'''
        self.items_list = sorted(self.items_list, key=lambda x: x.volume, reverse=False)
        self.items_list = sorted(self.items_list, key=lambda x: x.loadbear, reverse=True)
        # priority가 None이 아닌 아이템만 정렬
        self.items_list = sorted(
            self.items_list,
            key=lambda x: x.priority if x.priority is not None else 0,
            reverse=False
        )

    def get_simulation_result(self):
        candidates = [] 
        #self.current_bin.item_ids에서 self.bk_bin.item_ids를 뺀 나머지 아이템들을 candidates에 추가
        bk_item_ids = set(self.bk_bin.item_ids)
        for itm_id in self.current_bin.item_ids:
            if itm_id not in bk_item_ids:   # 이전 bin에 없던 아이템이면
                itm = global_item_manager.get(itm_id)
                if itm is not None:
                    # composition이면 그 아래 child를 candidates에 추가
                    if getattr(itm, "is_composite", False):
                        for cid in getattr(itm, "children_ids", []):
                            child_item = global_item_manager.get(cid)
                            candidates.append(child_item)
                    else:
                        candidates.append(itm)

        return candidates
    
    # =========== packing 관련 =============
    def pack(self, render: bool = False):
        '''
        packer의 bins에 item_list를 적재,
        packer의 item_list의 첫번째 아이템이 적재 가능한지 반환
        '''
        self.bk_bin = copy.deepcopy(self.current_bin)   #  target_item 고를 시 연산을 위해 적재 전 bin 저장

        if self.order_setting:
            self.orderItems()
            self.orderBins()

        '''
        self.packingModel.stack에 기대하는 것:
        하나의 bin에 최대한 많은 아이템을 적재하는 것.
        target_bin에 아이템을 적재-> current_bin._items와 current_bin._unfit_items에 아이템 추가됨. 각 리스트에 담긴 item class는 b_position에 위치 정보를 가지고 있음.
        '''
        if self.current_bin.size < 6:
            # 박스 크기를 큰 순으로 sort한뒤 제일 큰것만 남기고 삭제
            self.items_list = sorted(self.items_list, key=lambda x: x.volume, reverse=True)
            self.items_list = [self.items_list[0]]


        fit_result = self.packingModel.stack(self.current_bin, self.items_list)
        try_mode= False

        if render:
            self.current_bin.render(
                save_path=f'planning/renders/PalletFit_RL',
                name=f'{self.current_bin.size}_{self.current_bin.SU:.2f}_simulated',
                show=True,
                save=False,
                size_annotation = False,
                write_num = False,
                it_ids = [item._id for item in self.items_list],
            )

        if not(1 in fit_result): 
            fit_result = self.base_model1.stack(self.current_bin, self.items_list)
            try_mode= True

        if not(1 in fit_result):
            fit_result = self.base_model2.stack(self.current_bin, self.items_list)
            try_mode= True
        
        if 1 in fit_result:
            self.target_item = self.fix_and_clean_simulation(try_mode=try_mode)
        else:
            self.target_item = None
            self.unfit_items += self.current_bin.unfit_items

        return fit_result
    
    def fix_and_clean_simulation(self, try_mode=False):
        '''
        target_item 외의 아이템을 bin에서 제거
        '''
        self.log("🧹 [fix_and_clean_simulation] Cleaning simulation, keeping only target item...")

        if self.current_bin.size == 6 :
            print('debug')
        candidates = self.get_simulation_result()   # child만 리스트에 추가되어 반환

        if try_mode:
            selected_item = self._choose_target_item(candidates)
            self.target_item = selected_item

        elif self.model == "PalletFit_RL":
            self.target_item = self._choose_target_item_RL(candidates)
        else:
            self.target_item = self._choose_target_item(candidates)

        if not self.target_item:
            self.log("😭😭😭No target item found.😭😭😭")
            return
        print(f'self.target_item_id : {self.target_item._id}')

        self.log(f"🎉🎉🎉[[[[[Target item is: [{self.target_item.width}, {self.target_item.height}, {self.target_item.depth}], partno={self.target_item.partno}, id={self.target_item._id}, priority={self.target_item.priority}, b_position={self.target_item.b_position}, rotation={self.target_item.rotation_quat}]]]]]🎉🎉🎉")

        payload = self._make_gl_payload(self.target_item._id)
        if self.ui is not None:
            self.ui.render_signal.emit(payload)     # bytes 대신 dict 송신
        print(f'self.target_item_id : {self.target_item._id}')
        self.bins[self.current_bin_idx] = self.bk_bin  # 복제본으로 복원
        print(f'self.target_item_id : {self.target_item._id}')
        self._pushSimulation()
        print(f'self.target_item_id : {self.target_item._id}')
        
        self.current_bin.store(self.target_item)  # target_item만 다시 적재

        # Bin 단순화
        # self.current_bin.simplify()
        
        return self.target_item
    
    def _choose_target_item_RL(self, candidates):
        from utils.checkPivot import checkPivot_R
        """
        [RL 전용] PalletFit_RL 모델이 정한 순서(priority)가 가장 높은(낮은 숫자) 아이템을 선택한다.
        rl_adapter.py에서 stack() 수행 시, 배치 순서대로 item.priority = 1, 2, 3... 을 할당함.
        """
        if not candidates:
            return None

        # 1. priority가 설정된 아이템만 필터링 (None인 경우 제외)
        valid_candidates = [it for it in candidates if it.priority is not None]

        if not valid_candidates:
            self.log("[Packer] No items have priority set by RL. Fallback to first candidate.")
            return candidates[0]

        # 2. priority 오름차순 정렬 (1 -> 2 -> 3 ...)
        # 숫자가 작을수록 먼저 배치된(우선순위가 높은) 아이템
        valid_candidates.sort(key=lambda x: x.priority)

        # 3. 가장 우선순위 높은 아이템 선택
        best_item = valid_candidates[0]

        '''
        best_item.b_position의 [0,1]값(x,y)값이 x=0, y=0축에 self.current_bin.margin_x, margin_y 오차 이내에 
        있다면 축에 붙여서 놓을 수 있는지 확인하고 가능하면 0으로 변경하여 값 보정
        '''
        bx, by, bz = best_item.b_position
        mx, my = self.current_bin.margin_x, self.current_bin.margin_y
        
        if bx == 0 and by!= 0 and by <= my+EPS:
            check_bool, check_item = checkPivot_R(self.current_bin, best_item, [bx, 0, bz], best_item.rotation_quat, apply_margin=True)
            if check_bool:
                best_item.b_position[1] = 0
                global_item_manager.update(best_item._id, best_item)
        elif by == 0 and bx!= 0 and bx <= mx+EPS:
            check_bool, check_item = checkPivot_R(self.current_bin, best_item, [0, by, bz], best_item.rotation_quat, apply_margin=True)
            if check_bool:
                best_item.b_position[0] = 0
                global_item_manager.update(best_item._id, best_item)
        
        return best_item

    def _choose_target_item(self, candidates):

        '''
        적재 시뮬레이션 후보들 중 실제 적재할 아이템을 선택한다.

        1. b_position == [0,0,0] 인 아이템이 있다면, 그 중 가장 작은 volume을 가진 아이템을 즉시 선택.
        - 원점에 있는 아이템은 1개밖에 없다고 가정.
        2. 그렇지 않으면 전체 아이템을 z값 기준으로 정렬
        - z값이 낮을수록 우선
        - 같은 z값 내에서는 direction_overlap 기준으로 정렬
            (gap은 작고 ratio는 클수록 우선)
        3. 예외 처리: 선택된 아이템이 전체 중 가장 작은 volume이면, 다음 아이템을 대신 선택

        '''
        # 1) is_fixed == False 이고 composite가 아닌 아이템만 필터링
        if not candidates:
            return None
        
        if len(candidates) == 1:
            # 후보가 1개면 그 아이템을 선택
            self.prev_target_item_id = candidates[0]._id
            return candidates[0]

        # 1. 원점에 있는 아이템 우선 선택
        for it in candidates:
            if np.allclose(it.b_position, [0, 0, 0], atol=1e-6):
                self.prev_target_item_id = it._id
                return it  # 원점 아이템은 1개라고 가정
            
        # candidates 중에서 서로의 footprint가 겹치면 z값이 높은 것은 제외
        eps_area = 1e-6
        bounds_list = [aabb_bounds_xyz(it) for it in candidates]
        keep = [True] * len(candidates)

        for i in range(len(candidates)):
            if not keep[i]:
                continue
            bi = bounds_list[i]
            for j in range(len(candidates)):
                if i == j or not keep[j]:
                    continue
                bj = bounds_list[j]
                if compute_overlap_area(bi, bj) > eps_area:
                    # 같은 footprint 그룹이면 "가장 낮은 z_min"만 남김
                    if bj[4] < bi[4] - 1e-6:
                        keep[i] = False
                        break

        candidates = [it for it, k in zip(candidates, keep) if k]

        
        
        
        
        # 5. direction_overlap 기준 정렬
        def sort_key(it):
            get_direction_overlap(it, self.bk_bin)
            # update_child_positions(it)
            dg = it.options.get('direction_overlap', {})
            return (
                flat_surface_weight(self.bk_bin, it),
                (-dg['front'][3] + dg['left'][3]),  # overlap length
                (-dg['front'][1] + dg['left'][1]),
                it.ez,
            )
        sorted_items = sorted(candidates, key=sort_key)

        if not sorted_items:
            return None

        selected = sorted_items[0]
        return selected
    
    def check_binSize(self):
        '''
        self.current_bin의 각 꼭지점 8개에 해당하는 w_position으로 update_robot_state()을 보내서 반환값이 
        thresh_distance이상이면 Flase반환. 8점모두 이하이면 True를 반환하여
        해당 bin사이즈로 로봇이 singularity가 발생하지 않는지 확인하는 함수
        
        '''
        thresh = 0.005
        vers = self.current_bin.getVertices()
        for n, v in enumerate(vers[::-1]):
            v = [i*0.001 for i in v]    # mm ->m
            wp = shift_bin_coor2world(self.current_bin, v, RotationType.RT_WHD)
            print(f'check_binSize ] {n}번째 wp : {wp}')

            # wp로 robot 이동
            self.update_robot_state(
                nickname='binSize_test',
                cmd_type='movel',
                joints_des=self.joint_posori_dict['doosan_home']['joints'],
                posori_des = wp
            )
            gap = get_distance(wp, self.last_posori_state)
            print('check_binSize ] response : ', self.last_posori_state)
            print('check_binSize ] gap : ', gap)
            if gap < thresh: 
                self.update_robot_state(
                nickname='binSize_test',
                cmd_type='movej',
                joints_des=self.joint_posori_dict['doosan_home']['joints'],
                posori_des = wp
                )
                continue
            else:
                self.log(f'check_binSize ] {n}번째 wp로 이동 실패, gap: {gap} > {thresh}')

                return False

        return True
    
    def _pushSimulation(self):
        """
        target_item 중심에서 일정 거리(OFF mm)만큼 떨어진 8-방향에
        gripper를 넣을 수 있는지 검사.  
        OFF를 20 → 15 → … 순으로 줄여 가며
        가장 먼저 빈 위치를 찾으면 그 좌표를 반환한다.
        실패하면 기존 place_w_position 유지.
        """
        # 0) 아이템이 하나뿐이면 그대로
        if self.current_bin.size == 1:
            self.pre_position = self.place_w_position
            return self.pre_position

        # 1) ─ gripper를 target 크기로 맞춤 ───────────────────────────
        tgt = self.target_item
        tw, th, td          = tgt.getDimension()
        gr                  = self.gripper
        gr.width, gr.height , gr.depth = tw, th, td
        gr.rotation_quat    = tgt.rotation_quat
        global_item_manager.update(gr._id, gr)

        bin_       = self.bk_bin
        BW, BH, BD = bin_.width, bin_.height, bin_.depth
        tx, ty, tz = tgt.b_position             # target 모서리


        # 2) ─ helper ---------------------------------------------------
        def can_place_at(x: float, y: float, z: float) -> bool:
            # (a) bin 경계
            gr_w, gr_h, gr_d = gr.getDimension()

            mode = getattr(self, 'packing_mode', None)
            clr  = 20 if mode == 'palletizing' else 0
            if (x < -clr or y < -clr or z < -clr or
                x + gr_w  > BW + clr or
                y + gr_h > BH + clr):
                return False

            # (b) 충돌
            gap = 8
            ids = bin_.search_xyz(x+EPS, x+gr_w+gap,
                                y+EPS, y+gr_h+gap,
                                z+EPS, BD)
            return len(ids) == 0

        # 3) ─ 8-방향 offset 템플릿 -------------------------------------
        def build_offsets(off: float):
            dx = dy = off
            return [
                ("-right_back",  ( dx,  dy)),  # ↙(+y)
                ("-right",       ( dx,   0)),  # ←
                ("-back",        (  0,  dy)),  # ↓
                ("-right_front", ( dx, -dy)),  # ↖
                ("-front",       (  0, -dy)),  # ↑
                ("-left",        (-dx,   0)),  # →
                ("-left_front",  (-dx, -dy)),  # ↗
                ("-left_back",   (-dx,  dy)),  # ↘
            ]

        # 4) ─ 여러 OFFSET 값 순차 시도 -------------------------------
        for OFF in (25.0, 22.5, 20.0, 15.0, 12.5, 10.0):
            for name, (ox, oy) in build_offsets(OFF):
                cx, cy, cz = tx + ox, ty + oy, tz   # 수평 이동
                if can_place_at(cx, cy, cz):
                    self.log(f"✅ push 방향: {name}  (offset={OFF} mm) "
                                    f"→ ({cx:.1f}, {cy:.1f}, {cz:.1f})")
                    gr.b_position = [cx, cy, cz+20]      # 여유 10 mm
                    global_item_manager.update(gr._id, gr)
                    w_pos = Item_b2w_position(gr, self.current_bin)
                    self.pre_position = np.concatenate([w_pos[:3],
                                        self.place_w_position[3:7]])
                    print("@@ target_position: ", self.place_w_position ,"@@")
                    print("@@ pre_position from _pushSimulation: ", self.pre_position ,"@@")
                    return self.pre_position

        # 5) 전부 실패
        self.log("❌ 25~10 mm 모두 충돌 – push 불가, 원위치 유지")
        self.pre_position = self.place_w_position
        return self.pre_position
        
    def packingSimulation(self):
        '''
        item을 현재 bin에 적재할 수 있는지 시뮬레이션
        '''
        if len(self.bins) == 0:
            self.log("No bins available for packing simulation.")
            return None
        
        self.log("🖥 Starting packing simulation...🖥")
        self.items_list = self.conveyor_items # 인식된 아이템을 연산 대상에 반영
        packed_result = self.pack()  # 적재 연산

        if  self.has_target_item is False:
            self.can_packing = False
            items = self.current_bin.get_all_items()
            json_path = dump_items_to_json(items)             # 파일 생성

            self.log(f"packing 실패 – {json_path}에 덤프 완료")
        
        else:
            self.can_packing = True
            self.log(f'이번 연산에서 적재된 아이템 개수: {packed_result.count(1)}')

    # =========== decider 관련 =========
    @property
    def packing_complete(self) -> bool:
        '''
        1. bin_items가 max_items보다 크거나 같거나
        2. can_packing이 False일 때
        return True
        '''
        if self.current_bin is None:
            return False
        if self.can_packing is False:
            return True
        return self.bins[-1].size > self.max_items

    @property
    def has_target_item(self):
        return self.target_item is not None
    
    @property
    def target_w_position(self):
        if self.target_item is None:
            return None
        
        return self.target_item.w_position
    
    @property
    def is_attached(self):
        """
        Item의 options에 'is_attached' 값이 True이면 True 반환
        """
        return self.target_item is not None and self.target_item.options.get('is_attached', False)    
    
    def update_target_is_attached(self, is_attached:bool):
        if self.target_item is not None:
            self.target_item.options['is_attached'] = is_attached

            self.log(f'update_target_is_attached : {is_attached} 실행.')

        else:
            self.log('target_item이 없습니다.')

    def mark_target_item_as_completed(self):
        '''
        target_item을 bin_items에 추가하고, target_item을 None으로 설정합니다.
        '''
        self.conveyor_items.clear()
        self.target_item.options['is_fixed'] = True
        self.target_item.options['color'] = 'gray'
        self.update_target_is_attached(False)
        self.target_item = None

    def print_bin_states(self):
        '''
        현재 bin의 상태를 출력합니다.
        '''
        if self.current_bin is None:
            self.log("No current bin available.")
            return
        
        self.log(f"SU : {self.current_bin.SU}")
        self.log(f"count of items in bin: {self.current_bin.size}")

    # =========== UI rendering 관련 =============
    def _make_gl_payload(self, target_id=None):
        items = []
        for it in self.current_bin.get_all_items():
            w, h, d = it.getDimension()
            items.append(
                GLItemInfo(
                    id=it._id, x=it.b_position[0], y=it.b_position[1], z=it.b_position[2],
                    w=w, h=h, d=d,
                    color=it.options.get('color', 'gray'),
                    is_target=(it._id == target_id),
                )
            )
        # bin size도 같이
        meta = dict(width=self.current_bin.width,
                    height=self.current_bin.height,
                    depth=self.current_bin.depth,
                    name=self.current_bin.name)
        return dict(bin=meta,
                    items=[asdict(x) for x in items])
    

@dataclass
class GLItemInfo:
    id: int
    x: float; y: float; z: float      # b-좌표
    w: float; h: float; d: float
    color: str
    is_target: bool = False


