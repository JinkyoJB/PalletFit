# pt_dataset_to_items_json.py
import os, json
import torch

PT_PATH1 = "$PALLETFIT_ROOT/planning/RL/PCT/dataset/setting123_discrete.pt"
OUT_DIR1 = "$PALLETFIT_ROOT/planning/data/Item_data/paper/setting123_discrete"

PT_PATH2 = "$PALLETFIT_ROOT/planning/RL/PCT/dataset/setting2_continuous.pt"
OUT_DIR2 = "$PALLETFIT_ROOT/planning/data/Item_data/paper/setting2_continuous"

PT_PATH3 = "$PALLETFIT_ROOT/planning/RL/PCT/dataset/setting13_continuous.pt"
OUT_DIR3 = "$PALLETFIT_ROOT/planning/data/Item_data/paper/continuous_setting13"

PT_PATH4 = "$DRL_REPO/dataset/rs.pt"
OUT_DIR4 = "planning/data/Item_data/paper_drl/rs"

PT_PATH = PT_PATH4
OUT_DIR = OUT_DIR4

# 그리드 → mm 스케일 (필요하면 조정)
SCALE_X = 100.0
SCALE_Y = 100.0
SCALE_Z = 100.0

# 고정 필드 기본값
DEFAULT_ITEM = {
    "name": "",
    "objshape": "cube",
    "priority": 1,
    "updown": False,
    "weight": 0.84,
    "loadbear": 8.0,
    "unit": "mm",
    "rotation_quat": [0.0, 0.0, 0.0, 1.0],  # 데이터셋엔 회전 없음
}

def box_to_item(idx, box):
    # box: [x, y, z] 또는 [x, y, z, den]
    lst = box.tolist() if hasattr(box, "tolist") else list(box)
    x, y, z = map(float, lst[:3])
    den = float(lst[3]) if len(lst) > 3 else None

    item = {
        "partno": str(idx),
        **DEFAULT_ITEM,
        "width":  x * SCALE_X,
        "height": y * SCALE_Y,
        "depth":  z * SCALE_Z,
        "b_position": [0.0, 0.0, 0.0],  # 데이터셋에는 위치 정보가 없음
    }
    if den is not None:
        item["density"] = den  # 필요 시 키 이름 변경 가능
    return item

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)

    # 보통 리스트(에피소드 리스트) 구조. dict라면 적절한 키로 꺼내세요.
    episodes = None
    if isinstance(data, (list, tuple)):
        episodes = data
    elif isinstance(data, dict):
        # 추정 키 후보들 중 있는 것 사용
        for k in ["episodes", "data", "dataset", "test", "train"]:
            if k in data:
                episodes = data[k]
                break
        if episodes is None:
            # 알 수 없으면 한 에피소드로 간주
            episodes = [data]
    else:
        episodes = [data]

    for epi, ep in enumerate(episodes):
        items = [box_to_item(i, box) for i, box in enumerate(ep)]
        out_path = os.path.join(OUT_DIR, f"dataset_episode_{epi:03d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(items)} items → {out_path}")

if __name__ == "__main__":
    main()
