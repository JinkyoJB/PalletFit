from planning.packer import Packer
from planning.item import RotationType, Item
import time as pytime
from itertools import islice
from utils.util_functions import load_offline_data
import os
import csv
import glob
import numpy as np


def experiment1():
    item_num = 3
    generations = ["EDP+POST", "EDP", "CP", "EP", "EMS"]

    base_data_dir = "planning/data/Item_data/paper/customset"
    save_dir = "planning/experiment1_1265856_steps"

    os.makedirs(save_dir, exist_ok=True)

    stats_storage = {gen: {'packed': [], 'su': [], 'time': []} for gen in generations}

    csv_path = os.path.join(save_dir, 'experiment1_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Method', 'Packed_Count', 'Total_Items', 'SU', 'Time_Sec'])

    json_files = sorted(glob.glob(os.path.join(base_data_dir, "*.json")))

    if not json_files:
        print(f"경고: {base_data_dir} 경로에 json 파일이 없습니다.")
        return

    print(f"총 {len(json_files)}개의 파일에 대해 실험을 시작합니다.")

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"\n Processing File: {file_name}")

        for method_name in generations:
            print(f"   Method: {method_name} ... ", end="", flush=True)

            if "+POST" in method_name:
                use_simplify = True
                real_cands_option = method_name.replace("+POST", "")
            else:
                use_simplify = False
                real_cands_option = method_name

            item_data = load_offline_data(file_path)
            bk_item_count = len(item_data)
            conveyor_items = list(islice(item_data, item_num))

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

            while len(item_data) > 0:
                for it in conveyor_items:
                    packer.addItem(Item(**it))

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

            final_packed_count = packer.current_bin.size
            final_su = packer.current_bin.SU

            print(f"Done. (Packed: {final_packed_count}/{bk_item_count}, SU: {final_su:.4f})")

            stats_storage[method_name]['packed'].append(final_packed_count)
            stats_storage[method_name]['su'].append(final_su)
            stats_storage[method_name]['time'].append(elapsed_time)

            save_name = f"{os.path.splitext(file_name)[0]}_{final_packed_count}_{final_su:.4f}_{method_name}"
            packer.current_bin.render(
                save_path=save_dir,
                name=save_name,
                show=False,
                save=True,
                size_annotation=False,
                write_num=False,
            )

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    file_name,
                    method_name,
                    final_packed_count,
                    bk_item_count,
                    f"{final_su:.6f}",
                    f"{elapsed_time:.2f}",
                ])

    print("\n Calculating Statistics...")
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(['--- STATISTICS ---', '', '', '', '', ''])
        writer.writerow(['Method', 'Metric', 'Mean', 'StdDev', 'Min', 'Max'])

        for method in generations:
            data = stats_storage[method]

            if data['packed']:
                p_mean, p_std = np.mean(data['packed']), np.std(data['packed'])
                writer.writerow([method, 'Packed_Count', f"{p_mean:.4f}", f"{p_std:.4f}", np.min(data['packed']), np.max(data['packed'])])

                s_mean, s_std = np.mean(data['su']), np.std(data['su'])
                writer.writerow([method, 'SU', f"{s_mean:.4f}", f"{s_std:.4f}", f"{np.min(data['su']):.4f}", f"{np.max(data['su']):.4f}"])

                t_mean, t_std = np.mean(data['time']), np.std(data['time'])
                writer.writerow([method, 'Time_Sec', f"{t_mean:.4f}", f"{t_std:.4f}", f"{np.min(data['time']):.2f}", f"{np.max(data['time']):.2f}"])

                writer.writerow([])

    print(f"모든 실험 및 통계 저장이 완료되었습니다. 파일: {csv_path}")


def experiment1_1():
    import pandas as pd

    item_num = 1
    generations = ["EDP+POST", "EDP", "CP", "EP", "EMS"]

    base_data_dir = "planning/data/Item_data/paper/customset"
    save_dir = "planning/Performance_comparison/experiment1_1"

    os.makedirs(save_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(base_data_dir, "*.json")))

    if not json_files:
        print(f"경고: {base_data_dir} 경로에 json 파일이 없습니다.")
        return

    print(f"총 {len(json_files)}개의 파일에 대해 실험을 시작합니다.")

    all_experiment_results = []
    step_log_data = []

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"\n Processing File: {file_name}")

        for method_name in generations:
            print(f"   Method: {method_name} ... ", end="", flush=True)

            if "+POST" in method_name:
                use_simplify = True
                real_cands_option = method_name.replace("+POST", "")
            else:
                use_simplify = False
                real_cands_option = method_name

            item_data = load_offline_data(file_path)
            packed_volume = 0

            packer = Packer(
                problem='online',
                unfit_stop_setting=False,
                order_setting=True,
                model='PalletFit_RL',
                rotation_type=RotationType.BasicRotation,
                cands_option=real_cands_option,
            )
            packer.build_bin("experiment_RL")

            bin_w = packer.current_bin.width
            bin_l = packer.current_bin.height
            bin_h = packer.current_bin.depth
            total_bin_volume = bin_w * bin_l * bin_h

            start_total_time = pytime.time()
            step_count = 0

            while len(item_data) > 0:
                conveyor_items = list(islice(item_data, item_num))

                packer.items_list.clear()
                for it in conveyor_items:
                    packer.addItem(Item(**it))

                t0 = pytime.time()
                packer.pack(render=False, simplify_items=use_simplify)
                step_time = pytime.time() - t0
                step_count += 1

                current_cands_count = packer.current_bin.options.get('last_simulation_candidates', 0)

                step_log_data.append({
                    'file': file_name,
                    'method': method_name,
                    'step': step_count,
                    'candidates': current_cands_count,
                    'time': step_time,
                })

                if packer.target_item:
                    target_name = packer.target_item.name

                    for i, it in enumerate(item_data):
                        if it['name'] == target_name:
                            packed_volume += (it['width'] * it['height'] * it['depth'])
                            del item_data[i]
                            break

                    packer.mark_target_item_as_completed()
                else:
                    packer.current_bin.render(
                        show=False, save=True,
                        save_path=save_dir,
                        name=f"{file_name}_{method_name}_final",
                    )
                    break

            elapsed_time = pytime.time() - start_total_time
            final_su = (packed_volume / total_bin_volume) * 100 if total_bin_volume > 0 else 0

            print(f"Done! (Items: {step_count}, SU: {final_su:.2f}%, Time: {elapsed_time:.2f}s)")

            all_experiment_results.append({
                'file': file_name,
                'method': method_name,
                'packed_items': step_count,
                'space_utilization': final_su,
                'total_time': elapsed_time,
            })

    df_steps = pd.DataFrame(step_log_data)
    df_steps.to_csv(f"{save_dir}/experiment_results_steps.csv", index=False)

    df_summary = pd.DataFrame(all_experiment_results)
    df_summary.to_csv(f"{save_dir}/experiment_results_summary.csv", index=False)

    print("\n All experiments completed and saved.")
