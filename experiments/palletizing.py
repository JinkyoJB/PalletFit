from planning.packer import Packer
from planning.item import RotationType, Item
import time as pytime
from utils.util_functions import load_offline_data
from typing import List
import random

PALLET_SIZE = (1100, 1100, 1800)           # (W, D, H) [mm]
AREA_THRESHOLD = PALLET_SIZE[0] * PALLET_SIZE[1] * 1.5
EMOJI_SUCCESS = "💞"
EMOJI_FAIL = "😭"


def palletizing(
    model: str = "PalletFit",
    data_path: str = "planning/data/Item_data/skt/debuging_skt3.json",
    iter_seed: int = 42,
    rotation_type=RotationType.BasicRotation,
):
    all_items = load_offline_data(data_path)
    random.seed(iter_seed)
    random.shuffle(all_items)
    data_idx = 0
    print(f"총 원본 아이템: {len(all_items)}")

    selected_items: List[dict] = []
    acc_area = 0.0

    def item_area(item: dict) -> float:
        return item["width"] * item["height"]

    def refill_buffer():
        nonlocal data_idx, acc_area
        while acc_area < AREA_THRESHOLD and data_idx < len(all_items):
            itm = all_items[data_idx]
            selected_items.append(itm)
            acc_area += item_area(itm)
            data_idx += 1

    refill_buffer()

    packer = Packer(
        problem="online",
        unfit_stop_setting=False,
        order_setting=False,
        model=model,
        rotation_type=rotation_type,
    )
    packer.build_bin("SKT")

    packed_items: List[dict] = []
    step = 0

    while selected_items:
        step += 1
        packer.items_list.clear()
        for it in selected_items:
            packer.addItem(Item(**it))
        start_t = pytime.time()
        packer.log(f"[{step}] packing start ({len(selected_items)=})")
        packer.pack()
        packer.log(f"[{step}] packing end   ({pytime.time()-start_t:.2f}s)")

        target_item_obj = packer.target_item

        if target_item_obj:
            for i, it in enumerate(selected_items):
                if it["name"] == target_item_obj.name:
                    selected_items.pop(i)
                    acc_area -= item_area(it)
                    packed_items.append(it)
                    break

            packer.current_bin.render(
                save_path="planning/renders/palletizing/fit",
                name=f"{iter_seed}_{len(packed_items)}_{step}",
                show=False,
                it_id=target_item_obj._id if target_item_obj else None,
            )
            refill_buffer()
            continue

        print(f"{EMOJI_FAIL}  packing 실패 – target_item 반환 실패")
        packer.current_bin.render(
            save_path="planning/renders/palletizing/unfit",
            name=f"{iter_seed}_{len(packed_items)}_{step}",
            show=False,
            it_id=target_item_obj._id if target_item_obj else None,
        )
        return

    print(f"{EMOJI_SUCCESS}  모든 아이템 적재 완료! {len(packed_items)}개 / {len(all_items)}개\n")


def palletizing2(
    model: str = "PalletFit",
    data_path: str = "planning/data/Item_data/skt/non-guillotine.json",
    iter_seed: int = 42,
    rotation_type=RotationType.BasicRotation,
):
    all_items = load_offline_data(data_path)
    print(f"총 원본 아이템: {len(all_items)}")

    packer = Packer(
        problem="online",
        unfit_stop_setting=False,
        order_setting=False,
        model=model,
        rotation_type=rotation_type,
    )
    packer.build_bin("SKT")

    packed_items: List[dict] = []
    step = 0

    for raw_it in all_items:
        step += 1
        packer.items_list.clear()
        packer.addItem(Item(**raw_it))

        start_t = pytime.time()
        packer.log(f"[{step}] packing start (item='{raw_it['name']}')")
        packer.pack()
        packer.log(f"[{step}] packing end   ({pytime.time()-start_t:.2f}s)")

        target_it = packer.target_item

        if target_it:
            packed_items.append(raw_it)
            packer.current_bin.render(
                save_path="planning/renders/palletizing/fit",
                name=f"{iter_seed}_{len(packed_items)}_{step}",
                show=False,
                it_id=target_it._id if target_it else None,
            )
            continue

        print(f"{EMOJI_FAIL}  '{raw_it['name']}' 적재 실패, 루프 종료")
        packer.current_bin.render(
            save_path="planning/renders/palletizing/unfit",
            name=f"{iter_seed}_{len(packed_items)}_{step}",
            it_id=target_it._id if target_it else None,
            show=False,
        )
        break

    print(f"{EMOJI_SUCCESS}  적재 완료: {len(packed_items)}/{len(all_items)}개 배치")
