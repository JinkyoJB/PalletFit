from planning.packer import Packer
from planning.item import RotationType
import time as pytime


def get_test(problem_type, model):
    packer = Packer(
        problem=problem_type,
        bin_path='planning/data/Bin_data/Margin_bin.json',
        recorded_path='planning/data/Item_data/trainset/single_case_8000ea_60.json',
        type_sampled_path='planning/data/trainset.json_bin450450200_seed20250.json',
        direct_item_list='planning/data/Item_data/trainset/RectangularPrism_100ea_2.json',
        unfit_stop_setting=True,
        order_setting=False,
        model=model,
        rotation_type=RotationType.BasicRotation,
    )

    start_time = pytime.time()
    packer.pack()
    print(f'걸린 시간: {pytime.time() - start_time}')

    for idx_bin, bin in enumerate(packer.bins):
        print(f'Bin name:{bin.name}, partno:{bin.partno}')
        if bin.unfit_items:
            for idx_unfit, unfit_item in enumerate(bin.unfit_items):
                print(
                    f'    Unfit item name:{unfit_item.name}, '
                    f'partno:{unfit_item.partno}, '
                    f'item_size:{unfit_item.width}x{unfit_item.height}x{unfit_item.depth}'
                )
        print('     packed items:')
        for item in bin.get_all_items():
            print(f'    Item name:{item.name}, partno:{item.partno}, position:{item.getResult()}')
        SU = bin.getSU()
        print('leftover volume:', SU, '%')
        print(f'적재된 아이템개수: {bin.size}')

    packer.packingModel.env.save_gif(plot_name=f'{item.width}x{item.height}x{item.depth}_SU{SU}')
