import torch
import argparse
import os
import numpy as np
import sys

def inspect_dataset(file_path, show_all=False):
    print(f"\n🔍 [Inspect] Loading file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    try:
        # [수정됨] weights_only=False 옵션 추가 (Numpy/List 데이터 로드 허용)
        data = torch.load(file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Error loading .pt file: {e}")
        return

    print(f"✅ Load successful!")
    print(f"=" * 60)
    print(f"📂 Data Type: {type(data)}")

    # 1. 리스트 형태 (PCT 데이터셋 표준 구조)
    if isinstance(data, list):
        num_episodes = len(data)
        print(f"📊 Total Episodes: {num_episodes}")
        
        if num_episodes == 0:
            print("⚠️ Warning: Dataset is empty.")
            return

        # --- [통계 분석] ---
        # 전체 데이터를 순회하며 Min/Max 및 차원 확인
        all_w, all_h, all_d = [], [], []
        dims_set = set()
        
        for ep in data:
            for item in ep:
                # 튜플/리스트 등 처리
                item_list = list(item)
                dims_set.add(len(item_list))
                if len(item_list) >= 3:
                    all_w.append(item_list[0])
                    all_h.append(item_list[1])
                    all_d.append(item_list[2])

        print(f"\n📏 [Scale & Dimension Analysis]")
        print(f"   - Dimensions detected per item: {list(dims_set)}")
        if 4 in dims_set:
            print("     👉 (w, h, d, density) -> Setting 3 (Correct)")
        elif 3 in dims_set:
            print("     👉 (w, h, d) -> Setting 1 or 2")
        
        if all_w:
            max_val = max(max(all_w), max(all_h), max(all_d))
            min_val = min(min(all_w), min(all_h), min(all_d))
            print(f"   - Value Range: Min {min_val} ~ Max {max_val}")
            
            if max_val <= 10.0:
                print("     ✅ Scale looks like PCT Discrete (0~10) or Continuous (0~1).")
            else:
                print("     ⚠️ Scale is LARGE (>10). Packer Scale (mm)? -> Needs Scaling!")

        print(f"=" * 60)

        # --- [출력 모드 분기] ---
        if show_all:
            print(f"📢 Printing ALL data ({num_episodes} episodes)...\n")
            for i, ep in enumerate(data):
                print(f"--- Episode {i} (Items: {len(ep)}) ---")
                for j, item in enumerate(ep):
                    print(f"  Item {j}: {item}")
        else:
            print(f"📢 Printing SAMPLE data (First 2 episodes). Use --all to see everything.\n")
            # 최대 2개 에피소드, 에피소드 당 최대 5개 아이템만 출력
            for i, ep in enumerate(data[:2]):
                print(f"--- Episode {i} (Total Items: {len(ep)}) ---")
                for j, item in enumerate(ep[:5]):
                    print(f"  Item {j}: {item}")
                if len(ep) > 5:
                    print(f"  ... (and {len(ep)-5} more items)")
            
            if num_episodes > 2:
                print(f"\n... (and {num_episodes - 2} more episodes)")

    # 2. 딕셔너리 형태 (가중치 파일)
    elif isinstance(data, dict):
        print(f"📂 Dictionary Keys: {list(data.keys())}")
        if show_all:
            print(data)
    
    # 3. 텐서 형태
    elif isinstance(data, torch.Tensor):
        print(f"SHAPE: {data.shape}")
        if show_all:
            print(data)
        else:
            print(data.flatten()[:20]) # 일부만 출력

    print(f"=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect .pt dataset file")
    parser.add_argument("path", type=str, nargs='?', 
                        default="dataset/my_testset.pt", 
                        help="Path to the .pt file")
    
    # --all 옵션 추가
    parser.add_argument("--all", action="store_true", 
                        help="Print ALL data content (default: sample only)")
    
    args = parser.parse_args()

    inspect_dataset(args.path, args.all)