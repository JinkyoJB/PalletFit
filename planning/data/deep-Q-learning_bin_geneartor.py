import json

def create_bin_json(matrix_resolution, min_bin, margin_x, margin_y, unit, output_filename="bin.json"):
    """
    matrix_resolution: [x_cells, y_cells, z_cells]
    min_bin: [min_width, min_height, min_depth]
    margin_x: 가로 아이템 간 간격
    margin_y: 세로 아이템 간 간격
    unit: 단위 (예: 'mm')
    output_filename: 출력할 JSON 파일 이름
    """
    
    # 계산:
    # 최소 사이즈에 (셀 수 - 1) * margin만큼 추가
    width  = min_bin[0] + (matrix_resolution[0] - 1) * margin_x
    height = min_bin[1] + (matrix_resolution[1] - 1) * margin_y
    depth  = min_bin[2]  # z방향 마진은 별도 계산 조건이 없으므로 그대로 사용
    
    bin_data = [
        {
            "partno": "BIN_0",
            "name": "Bin_0",
            "width": width,
            "height": height,
            "depth": depth,
            "max_weight": 2000,
            "w_position": None,
            "margin_x": margin_x,
            "margin_y": margin_y,
            "options": {
                "min_bin": min_bin
            },
            "unit": unit,
            "corner": 0,
            "support_surface_ratio": 0.5
        }
    ]
    
    # JSON 파일로 저장
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(bin_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # 예시 입력값
    matrix_resolution = [20, 20, 20]
    min_bin = [400, 400, 400]
    margin_x = 5
    margin_y = 5
    unit = "mm"
    
    # 함수 호출
    create_bin_json(matrix_resolution, min_bin, margin_x, margin_y, unit, "planning/data/dqn-bin.json")
    print("bin.json 파일이 생성되었습니다.")
