# planning/main.py

from planning.packer import Packer
from planning.item import RotationType
import time as pytime
from itertools import islice
from utils.util_functions import *
import torch.cuda as cuda
from torch.cuda import empty_cache,  synchronize
from typing import List
from pathlib import Path

import glob, json, random, time as pytime
# import matplotlib.pyplot as plt
# import numpy as np


def experiment1():
    # ─────────────────────────────────────────────────────────────
    # 1. 설정 및 준비
    # ─────────────────────────────────────────────────────────────
    item_num = 3
    # "EDP+POST" 추가
    generations = ["EDP+POST", "EDP", "CP", "EP", "EMS"]
    
    base_data_dir = "planning/data/Item_data/paper/customset"
    save_dir = "planning/experiment1_1265856_steps"
    
    # 결과 저장 폴더 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 통계 집계용 딕셔너리 초기화
    stats_storage = {gen: {'packed': [], 'su': [], 'time': []} for gen in generations}
    
    # CSV 파일 초기화
    csv_path = os.path.join(save_dir, 'experiment1_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Method', 'Packed_Count', 'Total_Items', 'SU', 'Time_Sec'])

    json_files = sorted(glob.glob(os.path.join(base_data_dir, "*.json")))
    
    if not json_files:
        print(f"⚠️ 경고: {base_data_dir} 경로에 json 파일이 없습니다.")
        return

    print(f"🚀 총 {len(json_files)}개의 파일에 대해 실험을 시작합니다.")

    # ─────────────────────────────────────────────────────────────
    # 2. 실험 루프
    # ─────────────────────────────────────────────────────────────
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"\n📂 Processing File: {file_name}")

        for method_name in generations:
            print(f"   ▶ Method: {method_name} ... ", end="", flush=True)
            
            # [설정 분기] "+POST" 포함 여부에 따라 옵션 설정
            if "+POST" in method_name:
                use_simplify = True
                # Packer에는 원래 옵션 이름("EDP")만 전달
                real_cands_option = method_name.replace("+POST", "")
            else:
                use_simplify = False
                real_cands_option = method_name

            # 데이터 로드
            item_data = load_offline_data(file_path)
            bk_item_count = len(item_data)
            conveyor_items = list(islice(item_data, item_num))
            
            # Packer 초기화 (real_cands_option 사용)
            packer = Packer(
                problem='online',
                unfit_stop_setting=False,
                order_setting=True,
                model='PalletFit_RL',
                rotation_type=RotationType.BasicRotation,
                cands_option=real_cands_option,
            )
            packer.build_bin("experiment_RL")
            
            start_total_time = pytime.time()

            # ─────────────────────────────────────────────────────────
            # 3. 패킹 시뮬레이션
            # ─────────────────────────────────────────────────────────
            while len(item_data) > 0:
                for it in conveyor_items:
                    packer.addItem(Item(**it))
                
                # [수정] simplify_items 인자 적용
                packer.pack(render=False, simplify_items=use_simplify)
                
                if packer.target_item:
                    for i, it in enumerate(item_data):
                        if it['name'] == packer.target_item.name:
                            del item_data[i]
                            break
                    packer.mark_target_item_as_completed()
                    conveyor_items = list(islice(item_data, item_num))
                    packer.items_list.clear()
                else:
                    break
            
            elapsed_time = pytime.time() - start_total_time
            
            # ─────────────────────────────────────────────────────────
            # 4. 결과 저장 (method_name 사용)
            # ─────────────────────────────────────────────────────────
            final_packed_count = packer.current_bin.size
            final_su = packer.current_bin.SU
            
            print(f"Done. (Packed: {final_packed_count}/{bk_item_count}, SU: {final_su:.4f})")

            # 통계 저장
            stats_storage[method_name]['packed'].append(final_packed_count)
            stats_storage[method_name]['su'].append(final_su)
            stats_storage[method_name]['time'].append(elapsed_time)

            # 렌더링 저장 (파일명에 method_name 포함)
            save_name = f"{os.path.splitext(file_name)[0]}_{final_packed_count}_{final_su:.4f}_{method_name}"
            packer.current_bin.render(
                save_path=save_dir,
                name=save_name,
                show=False,
                save=True,
                size_annotation=False,
                write_num=False,
            )

            # CSV 기록
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    file_name, 
                    method_name, 
                    final_packed_count, 
                    bk_item_count, 
                    f"{final_su:.6f}", 
                    f"{elapsed_time:.2f}"
                ])

    # ─────────────────────────────────────────────────────────────
    # 5. 최종 통계 계산
    # ─────────────────────────────────────────────────────────────
    print("\n📊 Calculating Statistics...")
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(['--- STATISTICS ---', '', '', '', '', ''])
        writer.writerow(['Method', 'Metric', 'Mean', 'StdDev', 'Min', 'Max'])

        for method in generations:
            data = stats_storage[method]
            
            if data['packed']:
                # Packed Count
                p_mean, p_std = np.mean(data['packed']), np.std(data['packed'])
                writer.writerow([method, 'Packed_Count', f"{p_mean:.4f}", f"{p_std:.4f}", np.min(data['packed']), np.max(data['packed'])])
                
                # SU
                s_mean, s_std = np.mean(data['su']), np.std(data['su'])
                writer.writerow([method, 'SU', f"{s_mean:.4f}", f"{s_std:.4f}", f"{np.min(data['su']):.4f}", f"{np.max(data['su']):.4f}"])
                
                # Time
                t_mean, t_std = np.mean(data['time']), np.std(data['time'])
                writer.writerow([method, 'Time_Sec', f"{t_mean:.4f}", f"{t_std:.4f}", f"{np.min(data['time']):.2f}", f"{np.max(data['time']):.2f}"])
                
                writer.writerow([])

    print(f"✨ 모든 실험 및 통계 저장이 완료되었습니다. 파일: {csv_path}")

def experiment1_1():
        import pandas as pd
        # ─────────────────────────────────────────────────────────────
        # 1. 설정 및 준비
        # ─────────────────────────────────────────────────────────────
        item_num = 1
        # "EDP+POST" 추가
        generations = ["EDP+POST", "EDP", "CP", "EP", "EMS"]
        
        base_data_dir = "planning/data/Item_data/paper/customset"
        save_dir = "planning/Performance_comparison/experiment1_1"
        
        # 결과 저장 폴더 생성
        os.makedirs(save_dir, exist_ok=True)

        json_files = sorted(glob.glob(os.path.join(base_data_dir, "*.json")))
        
        if not json_files:
            print(f"⚠️ 경고: {base_data_dir} 경로에 json 파일이 없습니다.")
            return

        print(f"🚀 총 {len(json_files)}개의 파일에 대해 실험을 시작합니다.")

        # ─────────────────────────────────────────────────────────────
        # 2. 실험 루프
        # ─────────────────────────────────────────────────────────────

        all_experiment_results = []
        step_log_data = [] # Fig 3 그래프(Step별 후보 수)를 그리기 위한 상세 로그

        for file_path in json_files:
            file_name = os.path.basename(file_path)
            print(f"\n📂 Processing File: {file_name}")

            for method_name in generations:
                print(f"   ▶ Method: {method_name} ... ", end="", flush=True)
                
                # [설정 분기] "+POST" 포함 여부에 따라 옵션 설정
                if "+POST" in method_name:
                    use_simplify = True
                    real_cands_option = method_name.replace("+POST", "")
                else:
                    use_simplify = False
                    real_cands_option = method_name

                # [리뷰어 방어용] 만약 각 방법론마다 별도로 학습된 모델이 있다면 경로를 분기 처리
                # model_checkpoint = f"./checkpoints/{real_cands_option}_best.pth" 
                
                # 데이터 로드
                item_data = load_offline_data(file_path)
                total_bin_volume = 0 # (Bin의 크기를 안다면 여기서 설정, 모르면 packer에서 가져옴)
                packed_volume = 0
                
                # Packer 초기화
                packer = Packer(
                    problem='online',
                    unfit_stop_setting=False,
                    order_setting=True,
                    model='PalletFit_RL', # 혹은 model_path=model_checkpoint
                    rotation_type=RotationType.BasicRotation,
                    cands_option=real_cands_option,
                )
                # bin 초기화 (여기서 bin size가 결정된다고 가정)
                packer.build_bin("experiment_RL")
                
                # Bin 부피 계산 (SU 계산용)
                bin_w, bin_l, bin_h = packer.current_bin.width, packer.current_bin.height, packer.current_bin.depth
                total_bin_volume = bin_w * bin_l * bin_h

                start_total_time = pytime.time()
                
                # 로깅 변수
                step_count = 0

                # ─────────────────────────────────────────────────────────
                # 3. 패킹 시뮬레이션
                # ─────────────────────────────────────────────────────────
                while len(item_data) > 0:
                    # 컨베이어 아이템 슬라이싱 (현재 남은 아이템 중에서)
                    conveyor_items = list(islice(item_data, item_num))
                    
                    # Packer에 아이템 등록
                    packer.items_list.clear() # 이전 스텝 잔여물 제거 (중요)
                    for it in conveyor_items:
                        packer.addItem(Item(**it))
                    
                    # ⏱️ 스텝별 실행 시간 측정 시작
                    t0 = pytime.time()
                    
                    # [핵심] pack 실행 (simplify 옵션 적용)
                    # pack 함수가 (성공여부, 정보) 등을 리턴하도록 수정되어 있다고 가정하거나
                    # packer 내부 상태를 조회해야 함
                    packer.pack(render=False, simplify_items=use_simplify)
                    
                    step_time = pytime.time() - t0
                    step_count += 1
                    
                    # [데이터 수집 1] 생성된 후보 수 가져오기 (Packer 코드 내부에 해당 변수가 있어야 함)
                    current_cands_count = packer.current_bin.options.get('last_simulation_candidates', 0)

                    # [데이터 수집 2] Step별 상세 로그 (Fig 3 그래프용)
                    step_log_data.append({
                        'file': file_name,
                        'method': method_name,
                        'step': step_count, # = Number of Items Packed
                        'candidates': current_cands_count,
                        'time': step_time
                    })

                    if packer.target_item:
                        # 적재 성공 시
                        target_name = packer.target_item.name
                        
                        # 원본 리스트에서 삭제 및 부피 계산
                        for i, it in enumerate(item_data):
                            if it['name'] == target_name:
                                # SU 계산을 위한 부피 누적
                                packed_volume += (it['width'] * it['height'] * it['depth'])
                                del item_data[i]
                                break
                        
                        packer.mark_target_item_as_completed()
                    else:
                        # 더 이상 적재 불가능
                        packer.current_bin.render(show=False, save=True, save_path=save_dir, name=f"{file_name}_{method_name}_final")
                        break
                
                elapsed_time = pytime.time() - start_total_time
                final_su = (packed_volume / total_bin_volume) * 100 if total_bin_volume > 0 else 0
                
                print(f"Done! (Items: {step_count}, SU: {final_su:.2f}%, Time: {elapsed_time:.2f}s)")

                # [데이터 수집 3] 에피소드(파일)별 최종 결과 (Table 1용)
                all_experiment_results.append({
                    'file': file_name,
                    'method': method_name,
                    'packed_items': step_count,
                    'space_utilization': final_su,
                    'total_time': elapsed_time,
                })

        # ─────────────────────────────────────────────────────────────
        # 4. 결과 저장 (CSV)
        # ─────────────────────────────────────────────────────────────
        # 그래프 그리기용 데이터 저장
        df_steps = pd.DataFrame(step_log_data)
        df_steps.to_csv(f"{save_dir}/experiment_results_steps.csv", index=False)

        # 표 만들기용 데이터 저장
        df_summary = pd.DataFrame(all_experiment_results)
        df_summary.to_csv(f"{save_dir}/experiment_results_summary.csv", index=False)

        print("\n✅ All experiments completed and saved.")
                    

def experiment2():
    # 1. 절대 경로 설정
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent 
    target_path = project_root / "data/Item_data/paper/customset" 
    
    # 파일 탐색
    all_json_files = sorted(target_path.glob("*.json"))
    if not all_json_files:
        raise FileNotFoundError(f"No JSON files found in {target_path}")

    # 🟢 [추가] 랜덤 샘플링 로직 (100개)
    # TARGET_SAMPLE_SIZE = 100
    
    # if len(all_json_files) > TARGET_SAMPLE_SIZE:
    #     # random.seed(42) # (선택사항) 실행할 때마다 똑같은 100개를 뽑고 싶다면 주석 해제
    #     json_files = random.sample(all_json_files, TARGET_SAMPLE_SIZE)
    #     json_files.sort()  # 보기 좋게 이름순 정렬
    #     print(f"🎲 Randomly sampled {TARGET_SAMPLE_SIZE} files from {len(all_json_files)} total files.")
    # else:
    #     json_files = all_json_files
    #     print(f"📂 Found {len(json_files)} files (less than {TARGET_SAMPLE_SIZE}), processing all.")

    # 결과 저장 디렉토리
    weight_name = 'best_model_20260107'
    result_dir = Path(f'planning/Performance_comparison/EXP2_results/{weight_name}_continuous2')
    result_dir.mkdir(parents=True, exist_ok=True)
    result_log_path = result_dir / 'result_log.txt'

    # 통계용 리스트
    all_packed_counts = []
    all_su_values = []

    for i, jf in enumerate(all_json_files, start=1):
        print(f"\n===== Processing {i}/{len(all_json_files)}: {jf.name} =====")
        
        item_data = load_offline_data(str(jf))
        initial_item_count = len(item_data)
        
        packer = Packer(
            problem="online",
            model="PalletFit_RL",
            rotation_type=RotationType.BasicRotation,
            unfit_stop_setting=True,
            order_setting=False,
        )
        packer.robot_plag = True
        packer.build_bin("experiment_RL")

        conveyor_items = list(islice(item_data, 1)) 
        
        # ===== 적재 루프 =====
        while len(item_data) > 0 and conveyor_items:
            for it in conveyor_items:
                packer.addItem(Item(**it))
            
            # Pack 실행
            start = pytime.time()
            packed_result = packer.pack()
            
            target_item = packer.target_item

            if target_item:
                target_item.options['is_fixed'] = True
                
                found = False
                for idx, it in enumerate(item_data):
                    if it['name'] == target_item.name:
                        del item_data[idx]
                        found = True
                        break
                
                if not found: break 

                conveyor_items = list(islice(item_data, 1))
                packer.items_list.clear()
            
            else:
                print(f"❌ [{jf.name}] 적재 실패! (현재 Bin Size: {packer.current_bin.size}/{initial_item_count}) -> 다음 파일로 이동")
                break 

        # ===== 결과 기록 (개별 파일 종료 후) =====
        current_su = packer.current_bin.SU
        packed_count = packer.current_bin.size
        
        # 리스트에 현재 결과 추가
        all_packed_counts.append(packed_count)
        all_su_values.append(current_su)
        
        # 현재까지의 누적 평균/표준편차 실시간 계산
        cur_packed_mean = np.mean(all_packed_counts)
        cur_packed_std = np.std(all_packed_counts)
        cur_su_mean = np.mean(all_su_values)
        cur_su_std = np.std(all_su_values)

        print(f"✅ [{jf.name}] 종료. Packed: {packed_count}, SU: {current_su:.4f}")
        print(f"📊 [Current Stats ({i}/{len(all_json_files)})] "
              f"Packed Mean: {cur_packed_mean:.2f} (±{cur_packed_std:.2f}), "
              f"SU Mean: {cur_su_mean:.4f} (±{cur_su_std:.4f})")

        # 렌더링 저장
        packer.current_bin.render(
            save_path=str(result_dir), 
            name=f'{jf.stem}_{packer.current_bin.size}_{packer.current_bin.getDensity():.2f}_simulated',
            show=False,
            save=True,
        )

        with open(result_log_path, 'a', encoding='utf-8') as f:
            f.write(f'{jf.stem}, packed={packed_count}, SU={current_su:.4f}\n')

    # ===== 최종 통계 =====
    if all_packed_counts:
        final_packed_mean = np.mean(all_packed_counts)
        final_packed_std = np.std(all_packed_counts)
        final_su_mean = np.mean(all_su_values)
        final_su_std = np.std(all_su_values)

        print("\n" + "="*40)
        print("🎉 [Final Summary] 🎉")
        print(f"Sample Size : {len(all_packed_counts)}")
        print(f"Packed Count: {final_packed_mean:.4f} ± {final_packed_std:.4f}")
        print(f"Space Util  : {final_su_mean:.4f} ± {final_su_std:.4f}")
        print("="*40)

        with open(result_log_path, 'a', encoding='utf-8') as f:
            f.write("--------------------------------------------------\n")
            f.write(f"Final Average (Sample N={len(all_packed_counts)}): "
                    f"packed_mean={final_packed_mean:.4f}, packed_std={final_packed_std:.4f}, "
                    f"SU_mean={final_su_mean:.4f}, SU_std={final_su_std:.4f}\n")


def main_simulation(model, problem):
    item_num =3
    _profile = []

    for iter in range(300,301):
        item_data = load_offline_data("planning/data/Item_data/debug/debug260108_201549.json")
        
        bk_item_count = len(item_data)
        print(f'아이템 개수: {bk_item_count}')

        conveyor_items = list(islice(item_data, item_num))  # 8개 아이템 추출
        # conveyor_items = list(islice(item_data, bk_item_count))  # 8개 아이템 추출
        packed_items = []

        packer = Packer(
            problem=problem,
            # bin_path='planning/data/Bin_data/exhibition_bin_margin2.json',
            unfit_stop_setting=False,
            order_setting=True,
            model=model,
            rotation_type=RotationType.BasicRotation,
        )
        
        packer.build_bin("default2")
        packer.robot_plag = True
        # packer.build_bin("SKT")

        while len(item_data) > 0:
            for it in conveyor_items:
                packer.addItem(Item(**it))  # 아이템 추가
                packer.log(f"input item : [{it['width']}, {it['height']}, {it['depth']}]")

            start = pytime.time()
            # packer.log(f'packing start {start_time}')
            packed_result = packer.pack(render =False)

            packer.log(f'packing end {pytime.time() - start} iter: {packer.current_bin.size},')
                # ==== ❷  평 가  후  ==================================
            _profile.append({
                "step":       packer.current_bin.size,
                "elapsed":    time.time() - start,
                "pivot_cnt":  getattr(packer.current_bin,"pivotTree.size", 0),  # pivotTree가 없을 수도 있으므로
            })
            packed_num = packed_result.count(1)
            target_item = packer.target_item

            packer.current_bin.render(
                save_path=f'planning/renders/PalletFit_RL/fit',
                name=f'{iter}_{packer.current_bin.size}_{packer.current_bin.SU:.2f}',
                show=True,
                save=False,
                # size_annotation = False,
                # write_num = False,
                it_ids=[target_item._id] if target_item else None,
            )
            print(f'적재된 아이템 개수: {packed_num}')

            # 3) candidates 정렬

            if target_item:
                target_item.options['is_fixed'] = True  # 시각화용

                # ✅ (6) item_list에서 target_item과 같은 name 가진 item을 1개만 삭제
                for i, it in enumerate(item_data):
                    if it['name'] == target_item.name:
                        packed_items.append(it)
                        del item_data[i]
                        break

                conveyor_items = list(islice(item_data, item_num))
            
            else:
                print('이런 일이 안일어났으면 좋겠다....')
                # 파일명 예: debugging26.json, debugging27.json ...
                debugging_file = f'planning/data/Item_data/exhibition/debugging{iter}.json'
                with open(debugging_file, 'w', encoding='utf-8') as f:
                    json.dump(item_data, f, indent=4, ensure_ascii=False)
                print(f'packing된 아이템 개수: {packer.current_bin.size}/{bk_item_count} , iter:{iter}💞')
                print(f'SU: {packer.current_bin.SU:.4f} , iter:{iter}💞')
                break
            
            # bin 후처리
            # packer.current_bin.post_merge(packer.target_item._id)
            print(f'packing된 아이템 개수: {packer.current_bin.size}/{bk_item_count} , iter:{iter}💞')
            packer.items_list.clear()  # 다음 루프를 위해 아이템 리스트 초기화

        else:
            # while 문이 break 없이 "정상 종료"되면 실행되는 블록
            packer.current_bin.render(
                save_path=f'planning/renders/PalletFit_RL',
                name=f'{iter}_{packer.current_bin.size}_{packer.current_bin.SU}_final',
                show=True,
                save=True,
                size_annotation = False,
                write_num = False,

            )

            print(f'packing된 아이템 개수: {packer.current_bin.size}/{bk_item_count} , iter:{iter}💞')
            print(f'SU: {packer.current_bin.SU:.4f} , iter:{iter}💞')







# ──────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────
PALLET_SIZE      = (1100, 1100, 1800)                # (W, D, H) [mm]
AREA_THRESHOLD   = PALLET_SIZE[0] * PALLET_SIZE[1] * 1.5   # W×D×1.5
EMOJI_SUCCESS    = "💞"
EMOJI_FAIL       = "😭"



def palletizing(
    model: str             = "PalletFit",
    data_path: str         = "planning/data/Item_data/skt/debuging_skt3.json",
    # data_path: str         = "planning/data/Item_data/skt/guillotine.json",
    iter_seed: int         = 42,
    rotation_type          = RotationType.BasicRotation,
):
    # 0) 데이터 준비 ───────────────────────────────────
    all_items = load_offline_data(data_path)
    random.seed(iter_seed)
    random.shuffle(all_items)
    data_idx = 0                               # 아직 꺼내지 않은 위치
    print(f"총 원본 아이템: {len(all_items)}")

    # 1) 버퍼(selected_items) & 누적 면적 초기화 ──────
    selected_items: List[dict] = []
    acc_area = 0.0
    
    def item_area(item: dict) -> float:
        """면적 = width × height (질문 기준)"""
        return item["width"] * item["height"]


    def refill_buffer():
        """selected_items의 누적 면적이 TH 미만이면 all_items에서 꺼내 채움"""
        nonlocal data_idx, acc_area
        while acc_area < AREA_THRESHOLD and data_idx < len(all_items):
            itm = all_items[data_idx]
            selected_items.append(itm)
            acc_area += item_area(itm)
            data_idx += 1

    refill_buffer()                             # 최초 채우기

    # 2) Packer 인스턴스 준비 ─────────────────────────
    packer = Packer(
        problem="online",
        unfit_stop_setting=False,
        order_setting=False,
        model=model,
        rotation_type=rotation_type,
    )
    packer.build_bin("SKT")                     # 팔레트 생성

    packed_items: List[dict] = []
    step = 0                                    # 반복 카운터

    # 3) 메인 적재 루프 ───────────────────────────────
    while selected_items:
        step += 1
        packer.items_list.clear()  # 이전 아이템 리스트 초기화
        # packer 는 Item 객체를 받으므로 변환
        for it in selected_items:
            packer.addItem(Item(**it))  # 아이템 추가
        start_t = pytime.time()
        packer.log(f"[{step}] packing start ({len(selected_items)=})")
        packer.pack()
        packer.log(f"[{step}] packing end   ({pytime.time()-start_t:.2f}s)")

        target_item_obj = packer.target_item   # 한 번에 한 개만!

        # ── 3-A. 성공: target_item 반환 시 ────────────
        if target_item_obj:
            # selected_items 에서 동일 name(또는 고유 id) 한 개 제거
            for i, it in enumerate(selected_items):
                if it["name"] == target_item_obj.name:
                    selected_items.pop(i)
                    acc_area -= item_area(it)
                    packed_items.append(it)
                    break

            # 성공 렌더(선택) ─ 원할 때만
            packer.current_bin.render(
                save_path="planning/renders/palletizing/fit",
                name=f"{iter_seed}_{len(packed_items)}_{step}",
                show=False,
                it_id=target_item_obj._id if target_item_obj else None,

                # pivotTree=packer.current_bin.pivotTree,
            )
            # 버퍼 다시 채우기
            refill_buffer()
            continue

        # ── 3-B. 실패: target 없음 → 렌더 후 종료 ─────
        print(f"{EMOJI_FAIL}  packing 실패 – target_item 반환 실패")
        packer.current_bin.render(
            save_path="planning/renders/palletizing/unfit",
            name=f"{iter_seed}_{len(packed_items)}_{step}",
            show=False,
            it_id=target_item_obj._id if target_item_obj else None,
        )
        return                                  # 조기 종료

    print(f"{EMOJI_SUCCESS}  모든 아이템 적재 완료! "
          f"{len(packed_items)}개 / {len(all_items)}개\n")
    
# ──────────────────────────────────────────────────────
# 메인 루프 – 아이템을 **하나씩** 넣어가며 적재
# ──────────────────────────────────────────────────────
def palletizing2(
    model: str             = "PalletFit",
    data_path: str         = "planning/data/Item_data/skt/non-guillotine.json",
    iter_seed: int         = 42,
    rotation_type          = RotationType.BasicRotation,
):
    # 0) 데이터 준비 ────────────────────────────────────
    all_items = load_offline_data(data_path)
    # random.seed(iter_seed)
    # random.shuffle(all_items)
    print(f"총 원본 아이템: {len(all_items)}")

    # 1) packer 초기 세팅 ───────────────────────────────
    packer = Packer(
        problem="online",
        unfit_stop_setting=False,
        order_setting=False,
        model=model,
        rotation_type=rotation_type,
    )
    packer.build_bin("SKT")                     # 팔레트 생성

    packed_items: List[dict] = []
    step = 0

    # 2) 아이템을 **순차적으로 하나씩** 시도 ----------------
    for raw_it in all_items:
        step += 1
        packer.items_list.clear()               # 이전 후보 리셋
        packer.addItem(Item(**raw_it))          # 이번에 딱 1개 넣기

        start_t = pytime.time()
        packer.log(f"[{step}] packing start (item='{raw_it['name']}')")
        packer.pack()
        packer.log(f"[{step}] packing end   ({pytime.time()-start_t:.2f}s)")

        target_it = packer.target_item    # 성공 시 Item 반환

        if target_it:                           # ─ 성공 ─
            packed_items.append(raw_it)

            # (선택) 결과 렌더 저장
            packer.current_bin.render(
                save_path="planning/renders/palletizing/fit",
                name=f"{iter_seed}_{len(packed_items)}_{step}",
                show=False,
                it_id=target_it._id if target_it else None,
            )
            continue                            # 다음 아이템 시도

        # ─ 실패 : 현재 아이템 적재 불가 ────────────────
        print(f"{EMOJI_FAIL}  '{raw_it['name']}' 적재 실패, 루프 종료")
        packer.current_bin.render(
            save_path="planning/renders/palletizing/unfit",
            name=f"{iter_seed}_{len(packed_items)}_{step}",
            it_id=target_it._id if target_it else None,
            show=False,
        )
        break                                   # 더 이상 진행하지 않음

    # 3) 최종 결과 --------------------------------------
    print(f"{EMOJI_SUCCESS}  적재 완료: {len(packed_items)}/{len(all_items)}개 배치")



def get_itemJsonFime_by_seed(seed):
    '''
    random seed로 섞은 item list를 json 파일로 저장
    '''
    random.seed(seed)
    packer = Packer(
        problem = problem_type,
        bin_path = 'planning/data/Bin_data/exhibition_bin_margin.json',    #DAN용
        offline_item_path = 'planning/data/Item_data/exhibition/real_object3.json',
        unfit_stop_setting = True,
        order_setting = False,
        model = model,
        rotation_type = RotationType.BasicRotation,
        )
    
    # (1) 아이템 리스트 셔플
    random.shuffle(packer.items_list)

    # (2) 새롭게 partno 재할당 (셔플 후 순서대로)
    for idx, item in enumerate(packer.items_list):
        item.partno = idx  # partno를 셔플된 순서로 재부여

    # (3) dict로 변환
    item_dicts = []
    for item in packer.items_list:
        item_dicts.append({
            "partno": str(item.partno),
            "name": item.name,
            "objshape": item.objshape,
            "width": float(item.width),
            "height": float(item.height),
            "depth": float(item.depth),
            "rotation_quat": item.posori[3:],  # 예: 0,1,2 등
            "priority": item.priority,
            "updown": item.updown,
            "weight": float(item.weight),
            "loadbear": item.loadbear,
            "unit": item.unit,
            "position": item.b_position,
        })

        # 예) exhibition_seed2_items.json 으로 저장
        with open(f"planning/data/Item_data/exhibition/exhibition_seed{seed}_items.json", "w") as f:
            json.dump(item_dicts, f, indent=4)
        
    print(f"[INFO] seed=2 아이템 리스트를 exhibition_seed{seed}_items.json 으로 저장 완료")



def get_test(problem_type, model):
        packer = Packer(
            problem = problem_type,
            # bin_path = 'planning/data/Bin_data/testset_bin_margin.json',
            bin_path = 'planning/data/Bin_data/Margin_bin.json',    #DAN용
            # bin_path = 'planning/data/Bin_data/exhibition_bin_margin.json',
            offline_item_path = 'planning/data/Item_data/trainset/single_case_8000ea_60.json',
            # offline_item_path = 'planning/data/Item_data/trainset/RectangularPrism_100ea.json',
            # offline_item_path = 'planning/data/Item_data/exhibition/random_5.json',
            online_item_type_path = 'planning/data/trainset.json_bin450450200_seed20250.json',
            direct_item_list = 'planning/data/Item_data/trainset/RectangularPrism_100ea_2.json',
            # bin_path = 'planning/data/dqn-bin.json',
            unfit_stop_setting = True,
            order_setting = False,
            model = model,
            rotation_type = RotationType.BasicRotation,
            )
        
        start_time = pytime.time()
        packer.pack()
        print(f'걸린 시간: {pytime.time() - start_time}')

        for idx_bin, bin in enumerate(packer.bins):
            print(f'Bin name:{bin.name}, partno:{bin.partno}')
            if bin.unfit_items:
                for idx_unfit, unfit_item in enumerate(bin.unfit_items):
                    print(f'    Unfit item name:{unfit_item.name}, partno:{unfit_item.partno}, item_size:{unfit_item.width}x{unfit_item.height}x{unfit_item.depth}')
            print('     packed items:')
            for item in bin.get_all_items():
                print(f'    Item name:{item.name}, partno:{item.partno}, position:{item.getResult()}')
            SU = bin.getSU()
            print('leftover volume:', SU , '%')
            print(f'적재된 아이템개수: {bin.size}')
        
        # 결과 시각화
        # packer.render(SU)
        packer.packingModel.env.save_gif(plot_name=f'{item.width}x{item.height}x{item.depth}_SU{SU}')

if __name__ == "__main__":


    problem_type = 'online'
    # model = 'OutlineFit'
    # model = 'BestFit'
    # model = "PalletFit"
    # model = "PalletFit_RL"
    # model = 'SafeFit'
    # model = 'SingleFit'
    # model = 'FFD_Z'
    # model = 'ZoneFit'
    model = 'DQN'
    # get_exhibition_case()
    # get_test(problem_type, model)
    # get_train(problem_type, model)
    # get_test(problem_type, model)
    # get_exhibition(problem_type, model)
    # x= 494
    # get_itemJsonFime_by_seed(x)
    # test_combination(x)


    main_simulation(model, problem_type)
    # palletizing2(model= model)
    # experiment2()
