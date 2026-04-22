import json
import os
import random


# m단위
def make_single_case(num_items=0, width=100, height=100, depth=100, unit='mm', dir='', nickname=''):
    item_list = []
    for i in range(num_items):
        item = {}
        item['partno'] = '0'
        item['name'] = 'item_' + str(i)
        item['objshape'] = 'cube'
        item['width'] = width
        item['height'] = height
        item['depth'] = depth
        item['rotation_quat'] = 0

        item['priority'] = 1
        item['updown'] = False

        # WHD 숫자의 맨 앞 자리만 가져와서 곱하여 weight 값으로 설정
        first_digit = int(str(get_first_digit(width)))

        weight = first_digit**3

        item['weight'] = weight
        item['loadbear'] = weight * 1000
        item['unit'] = unit

        item_list.append(item)

    os.makedirs(dir, exist_ok=True)  # 디렉토리가 없으면 생성
    with open(os.path.join(dir, f"single_case_{num_items}ea_{nickname}.json"), "w") as f:
        json.dump(item_list, f, indent=4)



# m단위
def make_single_case_noise(num_items=0, width=100, height=100, depth=100, unit='mm', dir='', nickname=''):
    item_list = []
    for i in range(num_items):
        item = {}
        item['partno'] = '0'
        item['name'] = 'item_' + str(i)
        item['objshape'] = 'cube'

        # 노이즈 범위: -1mm ~ +3mm
        noise_w = random.randint(-2, 2)
        noise_h = random.randint(-2, 2)
        noise_d = random.randint(-2, 2)

        # 치수에 노이즈 추가 (최소 1mm 보장)
        noisy_width = max(1, width + noise_w)
        noisy_height = max(1, height + noise_h)
        noisy_depth = max(1, depth + noise_d)

        item['width'] = noisy_width
        item['height'] = noisy_height
        item['depth'] = noisy_depth
        item['rotation_quat'] = 0

        item['priority'] = 1
        item['updown'] = False

        # WHD 숫자의 맨 앞 자리만 가져와서 곱하여 weight 값으로 설정
        first_digit = int(str(get_first_digit(noisy_width)))

        weight = first_digit**3

        item['weight'] = weight
        item['loadbear'] = weight * 1000
        item['unit'] = unit

        item_list.append(item)

    os.makedirs(dir, exist_ok=True)  # 디렉토리가 없으면 생성
    with open(os.path.join(dir, f"single_case_{num_items}ea_{nickname}.json"), "w") as f:
        json.dump(item_list, f, indent=4)




def get_first_digit(number):
    """
    주어진 숫자의 첫 번째 자리를 반환하는 함수입니다.
    """
    return int(str(number).replace(".", "")[0])


def make_complicated_case(num_items=0, dimension_choices=[], unit="mm", file_name=""):
    """
    Generates a list of items with dimensions randomly selected from predefined dimension sets.
    Each item's weight is calculated based on the first digits of its dimensions.
    The generated list is saved as a JSON file.

    Parameters:
    - num_items (int): Number of items to generate.
    - dimension_choices (list of lists): Predefined sets of dimensions [width, height, depth].
    - unit (str): Unit of measurement for dimensions (default: "mm").
    - file_name (str): Directory path where the JSON file will be saved.

    Returns:
    - None
    """
    if not dimension_choices:
        raise ValueError("dimension_choices cannot be empty. Provide a list of dimension tuples.")

    item_list = []

    for i in range(num_items):
        item = {}
        item["partno"] = i
        item["name"] = f"item_{i}"
        
        # Randomly select a dimension set
        selected_dimensions = random.choice(dimension_choices)
        width, height, depth = selected_dimensions
        
        # Determine objshape based on dimensions
        objshape = "cube"

        item["objshape"] = objshape
        
        item["level"] = 1
        item["updown"] = False
        item["width"] = width
        item["height"] = height
        item["depth"] = depth

        # Calculate weight based on the first digits of width, height, depth
        first_digit_w = get_first_digit(width)
        first_digit_h = get_first_digit(height)
        first_digit_d = get_first_digit(depth)

        weight = first_digit_w * first_digit_h * first_digit_d

        item["weight"] = weight
        item["loadbear"] = weight * 1000
        item["unit"] = unit

        item_list.append(item)

    # Save the item list to the JSON file
    with open(file_name, "w") as f:
        json.dump(item_list, f, indent=4)

    print(f"Generated {num_items} items and saved to {file_name}")


def make_cuboid_case(num_items=0,
                     dimension_base=20,
                     min_multiple=1,
                     max_multiple=10,
                     p_cube=0.3,
                     unit='mm',
                     file_name=''):
    """
    20의 배수를 활용하여 (정육면체 또는 직육면체) 아이템을 랜덤 생성.
    
    Args:
        num_items (int): 생성할 아이템 개수
        dimension_base (int): 기본 단위(예: 20)
        min_multiple (int): dimension_base에 곱할 최소 배수
        max_multiple (int): dimension_base에 곱할 최대 배수
        p_cube (float): 정육면체(cube)를 뽑을 확률 (0~1)
        unit (str): 치수 단위
        file_name (str): 결과를 저장할 JSON 파일 경로 (없으면 파일 저장 생략)
    
    Returns:
        item_list (list): 생성한 아이템 정보의 리스트 (dict 형태)
    """
    item_list = []
    for i in range(num_items):
        item = {}
        item["partno"] = i
        item["name"] = f"item_{i}"
        
        # 1) 정육면체(cube) 또는 직육면체(cuboid)를 뽑을지 결정
        if random.random() < p_cube:
            # 정육면체
            factor = random.randint(min_multiple, max_multiple)
            w = h = d = factor * dimension_base
        else:
            # 직육면체
            w_factor = random.randint(min_multiple, max_multiple)
            h_factor = random.randint(min_multiple, max_multiple)
            d_factor = random.randint(min_multiple, max_multiple)
            w = w_factor * dimension_base
            h = h_factor * dimension_base
            d = d_factor * dimension_base
        
        # 2) 아이템 정보 채우기
        item["objshape"] = 'cube'
        item["width"] = w
        item["height"] = h
        item["depth"] = d
        
        item["rotation_quat"] = 0
        item["priority"] = 1
        item["updown"] = False        
        # weight를 "가로/세로/높이의 첫 자리수"로부터 간단히 산출
        # (기존 예시 코드를 따른 방식)
        first_digit_w = int(str(w)[0])
        first_digit_h = int(str(h)[0])
        first_digit_d = int(str(d)[0])
        weight = first_digit_w * first_digit_h * first_digit_d

        item["weight"] = weight
        item["loadbear"] = weight * 1000
        item["unit"] = unit
        
        item_list.append(item)
    
    # 3) JSON 파일 저장 (file_name이 주어졌을 경우)
    if file_name:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(item_list, f, indent=4)
        print(f"[INFO] Generated {num_items} items and saved to {file_name}")
    else:
        print("[WARNING] No file_name provided. Returning item_list only (not saved).")
    
    return item_list


# 예시 사용
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    num = 20*20*20
    # make_single_case(
    #     num,
    #     width = 20,
    #     height = 20,
    #     depth = 20,
    #     unit="mm",
    #     dir="planning/data/Item_data/trainset/",
    #     nickname="20"
    # )

    # make_single_case_noise(
    #     100,
    #     width = 138,
    #     height = 80,
    #     depth = 35,
    #     unit="mm",
    #     dir="planning/data/Item_data/trainset/",
    #     nickname="cancho"
    # )

    # # 20의 배수인 직육면체 
    # predefined_dimensions = [
    #     [20, 40, 20],
    #     [40, 80, 40],
    #     [60, 40, 60],
    #     [80, 80, 80],
    #     [100, 100, 100],
    #     [160, 160, 160],
    #     [200, 200, 200],

    # ]

    # # # Parameters
    # # num_items = 50
    # unit = "mm"
    # file_name = f"planning/data/Item_data/trainset/random_case_8000ea.json"

    # # Generate the complicated case
    # make_complicated_case(
    #     num_items=num,
    #     dimension_choices=predefined_dimensions,
    #     unit=unit,
    #     file_name=file_name
    # )

    # 예: 아이템 50개, 20의 배수, 1~5 사이 랜덤, 40% 확률로 정육면체
    out_file = f"planning/data/Item_data/trainset/cuboid_case_seed{seed}.json"
    make_cuboid_case(
        num_items=50,
        dimension_base=20,
        min_multiple=1,
        max_multiple=5,
        p_cube=0.4,
        unit="mm",
        file_name=out_file
    )