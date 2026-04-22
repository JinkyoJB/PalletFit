from planning.packer import Packer
from planning.item import RotationType, Item
import time as pytime
from itertools import islice
from utils.util_functions import load_offline_data
import json


def main_simulation(model, problem):
    item_num = 3
    _profile = []

    for iter in range(300, 301):
        item_data = load_offline_data("planning/data/Item_data/paper/debug250627_224514.json")

        bk_item_count = len(item_data)
        print(f'아이템 개수: {bk_item_count}')

        conveyor_items = list(islice(item_data, item_num))
        packed_items = []

        packer = Packer(
            problem=problem,
            unfit_stop_setting=False,
            order_setting=True,
            model=model,
            rotation_type=RotationType.BasicRotation,
        )

        packer.build_bin("default2")
        packer.robot_plag = True

        while len(item_data) > 0:
            for it in conveyor_items:
                packer.addItem(Item(**it))
                packer.log(f"input item : [{it['width']}, {it['height']}, {it['depth']}]")

            start = pytime.time()
            packed_result = packer.pack(render=False)

            packer.log(f'packing end {pytime.time() - start} iter: {packer.current_bin.size},')
            _profile.append({
                "step":      packer.current_bin.size,
                "elapsed":   pytime.time() - start,
                "pivot_cnt": getattr(packer.current_bin, "pivotTree.size", 0),
            })
            packed_num = packed_result.count(1)
            target_item = packer.target_item

            packer.current_bin.render(
                save_path=f'planning/renders/PalletFit_RL/fit',
                name=f'{iter}_{packer.current_bin.size}_{packer.current_bin.SU:.2f}',
                show=True,
                save=False,
                it_ids=[target_item._id] if target_item else None,
            )
            print(f'적재된 아이템 개수: {packed_num}')

            if target_item:
                target_item.options['is_fixed'] = True

                for i, it in enumerate(item_data):
                    if it['name'] == target_item.name:
                        packed_items.append(it)
                        del item_data[i]
                        break

                conveyor_items = list(islice(item_data, item_num))

            else:
                print('이런 일이 안일어났으면 좋겠다....')
                debugging_file = f'planning/data/Item_data/exhibition/debugging{iter}.json'
                with open(debugging_file, 'w', encoding='utf-8') as f:
                    json.dump(item_data, f, indent=4, ensure_ascii=False)
                print(f'packing된 아이템 개수: {packer.current_bin.size}/{bk_item_count} , iter:{iter}')
                print(f'SU: {packer.current_bin.SU:.4f} , iter:{iter}')
                break

            print(f'packing된 아이템 개수: {packer.current_bin.size}/{bk_item_count} , iter:{iter}')
            packer.items_list.clear()

        else:
            packer.current_bin.render(
                save_path=f'planning/renders/PalletFit_RL',
                name=f'{iter}_{packer.current_bin.size}_{packer.current_bin.SU}_final',
                show=True,
                save=True,
                size_annotation=False,
                write_num=False,
            )

            print(f'packing된 아이템 개수: {packer.current_bin.size}/{bk_item_count} , iter:{iter}')
            print(f'SU: {packer.current_bin.SU:.4f} , iter:{iter}')
