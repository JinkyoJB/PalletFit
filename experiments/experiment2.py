from planning.packer import Packer
from planning.item import RotationType, Item
import time as pytime
from itertools import islice
from utils.util_functions import load_offline_data
from pathlib import Path
import numpy as np


def experiment2():
    project_root = Path(__file__).resolve().parent.parent  # experiments/ -> repo root
    target_path = project_root / "planning/data/Item_data/paper/customset"

    all_json_files = sorted(target_path.glob("*.json"))
    if not all_json_files:
        raise FileNotFoundError(f"No JSON files found in {target_path}")

    weight_name = 'best_model_20260107'
    result_dir = Path(f'planning/Performance_comparison/EXP2_results/{weight_name}_continuous2')
    result_dir.mkdir(parents=True, exist_ok=True)
    result_log_path = result_dir / 'result_log.txt'

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

        while len(item_data) > 0 and conveyor_items:
            for it in conveyor_items:
                packer.addItem(Item(**it))

            packer.pack()

            target_item = packer.target_item

            if target_item:
                target_item.options['is_fixed'] = True

                found = False
                for idx, it in enumerate(item_data):
                    if it['name'] == target_item.name:
                        del item_data[idx]
                        found = True
                        break

                if not found:
                    break

                conveyor_items = list(islice(item_data, 1))
                packer.items_list.clear()

            else:
                print(
                    f"[{jf.name}] 적재 실패! "
                    f"(현재 Bin Size: {packer.current_bin.size}/{initial_item_count}) "
                    f"-> 다음 파일로 이동"
                )
                break

        current_su = packer.current_bin.SU
        packed_count = packer.current_bin.size

        all_packed_counts.append(packed_count)
        all_su_values.append(current_su)

        cur_packed_mean = np.mean(all_packed_counts)
        cur_packed_std = np.std(all_packed_counts)
        cur_su_mean = np.mean(all_su_values)
        cur_su_std = np.std(all_su_values)

        print(f"[{jf.name}] 종료. Packed: {packed_count}, SU: {current_su:.4f}")
        print(
            f"[Current Stats ({i}/{len(all_json_files)})] "
            f"Packed Mean: {cur_packed_mean:.2f} (±{cur_packed_std:.2f}), "
            f"SU Mean: {cur_su_mean:.4f} (±{cur_su_std:.4f})"
        )

        packer.current_bin.render(
            save_path=str(result_dir),
            name=f'{jf.stem}_{packer.current_bin.size}_{packer.current_bin.getDensity():.2f}_simulated',
            show=False,
            save=True,
        )

        with open(result_log_path, 'a', encoding='utf-8') as f:
            f.write(f'{jf.stem}, packed={packed_count}, SU={current_su:.4f}\n')

    if all_packed_counts:
        final_packed_mean = np.mean(all_packed_counts)
        final_packed_std = np.std(all_packed_counts)
        final_su_mean = np.mean(all_su_values)
        final_su_std = np.std(all_su_values)

        print("\n" + "=" * 40)
        print("[Final Summary]")
        print(f"Sample Size : {len(all_packed_counts)}")
        print(f"Packed Count: {final_packed_mean:.4f} ± {final_packed_std:.4f}")
        print(f"Space Util  : {final_su_mean:.4f} ± {final_su_std:.4f}")
        print("=" * 40)

        with open(result_log_path, 'a', encoding='utf-8') as f:
            f.write("--------------------------------------------------\n")
            f.write(
                f"Final Average (Sample N={len(all_packed_counts)}): "
                f"packed_mean={final_packed_mean:.4f}, packed_std={final_packed_std:.4f}, "
                f"SU_mean={final_su_mean:.4f}, SU_std={final_su_std:.4f}\n"
            )
