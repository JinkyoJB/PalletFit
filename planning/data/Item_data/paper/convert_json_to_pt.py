# convert_json_to_pt.py
import json
import torch
import glob
import os
import argparse
import random

def convert_dataset(input_dir, output_file, setting):
    """
    JSON 파일(mm 단위)을 읽어서 PCT 모델용 .pt 파일(0.0~1.0 단위)로 변환합니다.
    """
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not json_files:
        print(f"Error: No json files found in {input_dir}")
        return

    all_episodes = []
    print(f"Found {len(json_files)} json files. Converting for Setting {setting} (Continuous)...")

    # [핵심] 300mm -> 0.3으로 만들기 위해 1000으로 나눔
    SCALE_FACTOR = 100.0 

    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        
        episode_items = []
        for item in data:
            # 1. 크기 추출 (키 이름 호환성 체크)
            raw_w = float(item.get('width', item.get('w', 0)))
            raw_h = float(item.get('height', item.get('h', 0)))
            raw_d = float(item.get('depth', item.get('d', 0)))

            # 2. 스케일링 (mm -> 0.x 비율)
            # PCT Continuous 환경은 Bin 크기가 1.0이므로, 아이템도 1.0 이하의 소수점이어야 함
            w = raw_w / SCALE_FACTOR
            h = raw_h / SCALE_FACTOR
            d = raw_d / SCALE_FACTOR
            
            # [수정됨] int()나 round()를 쓰지 않고 float 그대로 사용!
            # 혹시 모를 부동소수점 미세 오차만 방지 (선택 사항)
            w = float(f"{w:.6f}")
            h = float(f"{h:.6f}")
            d = float(f"{d:.6f}")

            # 3. 데이터 저장
            if setting == 3:
                # Density는 이미 0~1 사이 값이므로 그대로 사용
                density = float(item.get('density', random.uniform(0.1, 1.0)))
                episode_items.append((w, h, d, density))
            else:
                episode_items.append((w, h, d))
        
        all_episodes.append(episode_items)

    # 저장
    torch.save(all_episodes, output_file)
    print(f"Successfully saved {len(all_episodes)} episodes to {output_file}")
    
    # 검증 출력
    if len(all_episodes) > 0:
        first_item = all_episodes[0][0]
        print(f"Sample data (First item): {first_item}")
        # 예: (0.3, 0.4, 0.2, 0.919694) 가 나와야 함

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, 
                        default="$PALLETFIT_ROOT/planning/data/Item_data/paper/testset",
                        help="JSON 폴더 경로")
    parser.add_argument("--output_file", type=str, 
                        default="$PCT_REPO/dataset/my_testset.pt",
                        help="저장할 .pt 파일 경로")
    parser.add_argument("--setting", type=int, default=3, help="실험 세팅 번호 (1, 2, 3)")
    
    args = parser.parse_args()

    convert_dataset(args.input_dir, args.output_file, args.setting)