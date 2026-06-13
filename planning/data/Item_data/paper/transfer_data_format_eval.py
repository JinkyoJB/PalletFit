# save_trajs_to_json.py
import os, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from utils.env_paths import repo_path

# ===== 입력 경로 =====
TRJS_NPY = str(repo_path("planning", "RL", "PCT", "logs", "evaluation",
                         "jk-2025.08.25-11-33-44", "trajs.npy"))

# ===== 출력 경로 (요청 경로) =====
SAVE_DIR = str(repo_path("planning", "data", "Item_data", "paper"))

# ===== 스케일(그리드→mm) & 원점 보정 =====
SCALE_X = 1.0
SCALE_Y = 1.0
SCALE_Z = 1.0
OFFSET_X = 0.0
OFFSET_Y = 0.0
OFFSET_Z = 0.0

# ===== 기본 필드 =====
DEFAULT_NAME = ""
DEFAULT_OBJSHAPE = "cube"
DEFAULT_PRIORITY = 1
DEFAULT_UPDOWN = False
DEFAULT_WEIGHT = 0.84
DEFAULT_LOADBEAR = 8.0
DEFAULT_UNIT = "mm"
DEFAULT_ROT = [0.0, 0.0, 0.0, 1.0]  # 회전 정보 없으니 단위 사원수

def extract_packed(ep):
    if isinstance(ep, dict):
        for k in ["packed", "placements", "trajectory", "traj", "steps"]:
            if k in ep:
                return ep[k]
    return ep

def to_df(packed):
    cols7 = ["x","y","z","lx","ly","lz","bin"]
    cols8 = cols7 + ["rot"]
    if len(packed) and len(packed[0]) == 7:
        df = pd.DataFrame(packed, columns=cols7)
    elif len(packed) and len(packed[0]) == 8:
        df = pd.DataFrame(packed, columns=cols8)
    else:
        df = pd.DataFrame(packed)
    df.insert(0, "order", np.arange(len(df)))
    return df

def df_to_items(df):
    items = []
    for _, r in df.iterrows():
        width  = float(r["x"])  * SCALE_X
        height = float(r["y"])  * SCALE_Y
        depth  = float(r["z"])  * SCALE_Z
        bx = float(r["lx"]) * SCALE_X + OFFSET_X
        by = float(r["ly"]) * SCALE_Y + OFFSET_Y
        bz = float(r["lz"]) * SCALE_Z + OFFSET_Z

        rotation_quat = DEFAULT_ROT
        if "rot" in df.columns:
            # rot값 사원수 매핑이 필요하면 여기에 추가
            pass

        items.append({
            "partno": str(int(r["order"])),
            "name": DEFAULT_NAME,
            "objshape": DEFAULT_OBJSHAPE,
            "width":  width,
            "height": height,
            "depth":  depth,
            "rotation_quat": rotation_quat,
            "priority": DEFAULT_PRIORITY,
            "updown": DEFAULT_UPDOWN,
            "weight": DEFAULT_WEIGHT,
            "loadbear": DEFAULT_LOADBEAR,
            "unit": DEFAULT_UNIT,
            "b_position": [bx, by, bz],
        })
    return items

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    episodes = np.load(TRJS_NPY, allow_pickle=True).tolist()
    print(f"#episodes = {len(episodes)}")

    for i, ep in enumerate(episodes):
        dfi = to_df(extract_packed(ep))
        itemsi = df_to_items(dfi)
        out_path = os.path.join(SAVE_DIR, f"episode_{i:03d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(itemsi, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(itemsi)} items → {out_path}")

if __name__ == "__main__":
    main()
