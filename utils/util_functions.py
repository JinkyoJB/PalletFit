from time import time
import os, datetime, json, time, pathlib, csv
from collections import Counter
from itertools import permutations
from pathlib import Path
from planning.item import Item
import numpy as np
from typing import List, Dict, Union

def eps(v: float = 1e-6) -> float: return max(v, 1e-6)
def clip01(x: float) -> float: return float(np.clip(x, 0.0, 1.0))

class Log:
    """
    간단 dual logger  
      • 파일(log.txt)로 저장  
      • 콘솔에도 동시에 출력 (quiet=False 로 끄기)
    """
    def __init__(self, log_dir: str, quiet: bool = False):
        os.makedirs(log_dir, exist_ok=True)
        self.fp   = open(os.path.join(log_dir, "log.txt"), "a", encoding="utf-8")
        self.quiet = quiet

    def write(self, msg: str):
        ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
        line = ts + str(msg)
        if not self.quiet:           # 터미널에도 표시
            print(line)
        self.fp.write(line + "\n")
        self.fp.flush()              # 즉시 기록

    def close(self):
        self.fp.close()

def ela_t(start_t):
    ela_sec = time() - start_t
    if ela_sec < 60:
        ela = f'{ela_sec:.0f}s'
    elif 60 <= ela_sec < 60 * 60:
        ela = f'{ela_sec // 60:.0f}m {ela_sec % 60:.0f}s'
    else:
        ela_min = ela_sec // 60
        ela = f'{ela_min // 60:.0f}h {ela_sec % 60:.0f}m'
    return ela

def is_perfect_cuboid(bin):
    """
    주어진 Bin에 적재된 아이템들이 정확히 하나의 직육면체로 쌓였는지 판단.
    - 바운딩 박스 부피 vs 아이템 실제 부피 합
    - 아이템이 없는 경우 False
    """
    items = bin.items
    if not items:
        return False

    # 바운딩 박스 계산
    all_vertices = []
    for item in items:
        all_vertices.extend(item.getVertices())

    min_x = min(v[0] for v in all_vertices)
    max_x = max(v[0] for v in all_vertices)
    min_y = min(v[1] for v in all_vertices)
    max_y = max(v[1] for v in all_vertices)
    min_z = min(v[2] for v in all_vertices)
    max_z = max(v[2] for v in all_vertices)

    box_w = max_x - min_x
    box_h = max_y - min_y
    box_d = max_z - min_z

    bounding_volume = box_w * box_h * box_d

    # 아이템들 실제 부피 합
    # (Item 클래스에 volume 프로퍼티가 있으면 그걸 사용해도 됨)
    items_volume_sum = sum(item.width * item.height * item.depth for item in items)

    # float 오차 고려해서 비교
    if abs(bounding_volume - items_volume_sum) < 1e-6:
        return True
    else:
        return False
    

def get_latest_weight_file(base_dir ='planning/RL/DQN/weight'):
    directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # 디렉토리를 시간순으로 정렬 (디렉토리 이름에 time_stamp가 포함되어 있
    # 다고 가정)
    sorted_directories = sorted(directories, key=lambda x: x.split('_')[1], reverse=True)

    # 가장 최신 디렉토리 선택
    latest_directory = sorted_directories[0] if sorted_directories else None

    if latest_directory:
        weight_file = os.path.join(base_dir, latest_directory, 'model_best.pth')
        return weight_file    
    else:
        raise ValueError("No weight directories found.")
    

def generate_unique_item_permutations(a, b, c):
    """
    주어진 항목의 개수를 기반으로 모든 고유한 Item 순열을 메모리 효율적으로 생성하는 함수.

    Parameters:
    a (int): [1,1,1] 항목의 개수
    b (int): [2,1,1] 항목의 개수
    c (int): [2,2,1] 항목의 개수

    Returns:
    generator: 고유한 순열을 생성하는 제너레이터
    """
    # 항목 유형 정의: (width, height, depth, weight)
    item_types = {
        '1': (80, 80, 80, 1),  # [1,1,1]
        '2': (160, 80, 80, 2),  # [2,1,1]
        '3': (160, 160, 80, 3),  # [2,2,1]
    }

    # 각 항목 유형에 해당하는 속성 튜플을 리스트에 추가
    elements = []
    elements.extend([item_types['1']] * a)
    elements.extend([item_types['2']] * b)
    elements.extend([item_types['3']] * c)

    # Counter를 사용하여 중복된 항목 처리
    element_counter = Counter(elements)

    # 순열 생성 제너레이터
    def backtrack(path, counter, n):
        if len(path) == n:
            yield list(path)
            return
        for item in counter:
            if counter[item] > 0:
                counter[item] -= 1
                path.append(item)
                yield from backtrack(path, counter, n)
                path.pop()
                counter[item] += 1

    n = len(elements)
    for perm in backtrack([], element_counter, n):
        yield [
            Item(
                name=str(i + 1),
                width=item[0],
                height=item[1],
                depth=item[2],
                weight=item[3],
                rotation_quat=0,
                loadbear=1,
                priority=1,

                updown=False,
                b_position=[0, 0, 0],
                w_position=[0, 0, 0],
                unit='mm',
                options={}
            )
            for i, item in enumerate(perm)
        ]


def generate_unique_item_combinations(a, b, c):
    """
    주어진 항목의 개수를 기반으로 모든 고유한 Item 조합을 생성하는 함수.

    Parameters:
    a (int): [1,1,1] 항목의 개수
    b (int): [2,1,1] 항목의 개수
    c (int): [2,2,1] 항목의 개수

    Returns:
    generator: 고유한 조합을 생성하는 제너레이터
    """
    # 항목 유형 정의: (width, height, depth, weight)
    item_types = {
        '1': (80, 80, 80, 1),  # [1,1,1]
        '2': (160, 80, 80, 2),  # [2,1,1]
        '3': (160, 160, 80, 3),  # [2,2,1]
    }

    # 각 항목 유형에 해당하는 속성 튜플을 리스트에 추가
    elements = []
    elements.extend([item_types['1']] * a)
    elements.extend([item_types['2']] * b)
    elements.extend([item_types['3']] * c)

    # 중복 제거된 순열을 생성하는 제너레이터
    unique_combinations = set(permutations(elements))

    for perm in unique_combinations:
        perm_items = []
        for i, (width, height, depth, weight) in enumerate(perm, 1):
            calculated_weight = (width // 100) * (height // 100) * (depth // 100)
            item = Item(
                name=str(i),
                width=width,
                height=height,
                depth=depth,
                weight=calculated_weight,
                rotation_quat=0,
                loadbear=1,
                priority=1,
                updown=False,
                b_position=[0, 0, 0],
                w_position=[0, 0, 0],
                unit='mm',
                options={}
            )
            perm_items.append(item)
        yield perm_items


def safe_max(values, default=0):
    """
    values 안에 None이 있을 수 있으므로,
    None이 아닌 값만 모아서 max를 구하고,
    유효 값이 하나도 없으면 default 반환
    """
    if not values:  # 빈 리스트인 경우
        return default
    filtered = [v for v in values if v is not None]
    return max(filtered) if filtered else default

def safe_min(values, default=0):
    """
    values 안에 None이 있을 수 있으므로,
    None이 아닌 값만 모아서 min를 구하고,
    유효 값이 하나도 없으면 default 반환
    """
    if not values:  # 빈 리스트인 경우
        return default
    filtered = [v for v in values if v is not None]
    return min(filtered) if filtered else default

def safe_argmax(values, default_value=-999999):
    """
    values: list of numbers (일부가 None일 수 있음)
    default_value: None을 치환할 값 (기본 -999999)
    return: argmax 인덱스 (int)
    """
    if not values:
        return None  # 혹은 -1 등

    processed = [(x if x is not None else default_value) for x in values]
    return np.argmax(processed)

def load_offline_data(item_path: Union[str, Path]) -> List[Dict]:
    """
    offline_path로부터 items을 읽어온 뒤, dict 리스트로 반환.
    - JSON: 리스트(JSON array) 기대
    - TXT : 각 줄이 JSON 오브젝트면 파싱, 아니면 {"id": line}으로 감싸기
    - CSV : DictReader로 읽기
    각 항목에 기본값: b_position=[0,0,0], partno=인덱스
    """
    p = Path(item_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()
    with p.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            item_list = json.load(f)  # JSON 배열 기대
        elif suffix == ".txt":
            lines = [ln.strip() for ln in f if ln.strip()]
            # 줄별 JSON이면 파싱, 아니면 문자열을 감싸서 dict로 변환
            parsed = []
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    parsed.append(obj)
                except json.JSONDecodeError:
                    parsed.append({"id": ln})
            item_list = parsed
        elif suffix == ".csv":
            reader = csv.DictReader(f)
            item_list = list(reader)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

    if not isinstance(item_list, list):
        raise ValueError("Expected a list of items in the file.")

    # 공통 기본값 채우기
    for idx, item in enumerate(item_list):
        if not isinstance(item, dict):
            item = {"id": item}
            item_list[idx] = item
        item.setdefault("b_position", [0, 0, 0])
        item.setdefault("partno", idx)

    return item_list
    

def dump_items_to_json(items, dst_dir="planning/data/Item_data/debug"):
    """
    `items`(Iterable[Item]) → 지정 폴더에 time-stamp 이름의 json 파일 저장
    반환값: 생성된 파일 경로(str)
    """
    os.makedirs(dst_dir, exist_ok=True)

    # (1) Item → dict
    payload = []
    for it in items:
        d = dict(
            partno         = str(getattr(it, "partno", it.name)),
            name           = str(it.name),
            objshape       = str(it.objshape),
            width          = float(it.width),
            height         = float(it.height),
            depth          = float(it.depth),
            rotation_quat  = list(map(float, it.rotation_quat)),
            priority       = int(getattr(it, "priority", 1)),
            updown         = bool(getattr(it, "updown", False)),
            weight         = float(getattr(it, "weight", 0)),
            loadbear       = float(getattr(it, "loadbear", 0)),
            unit           = str(getattr(it, "unit", "mm")),
            b_position     = list(map(float, it.b_position)),
        )
        payload.append(d)

    # (2) 파일명:  YYMMDD_HHMMSS.json
    ts  = time.strftime("%y%m%d_%H%M%S")
    path = pathlib.Path(dst_dir) / f"debug{ts}.json"

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return str(path)

def str_norm(s: str) -> str:
    '''
    문자열 정규화 유틸 
    '''
    return str(s).replace('-', '_').replace(' ', '_').upper()