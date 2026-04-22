# planning/Performance_comparison/EXP2_results/Deeppack3D/baseline.py

'''
export DP3D_PATH="$HOME/study/DeepPack3D"
export PYTHONPATH="$DP3D_PATH:$DP3D_PATH/src:$PYTHONPATH"

'''
from planning.packer import Packer
from planning.item import Item, RotationType
from utils.util_functions import load_offline_data
from pathlib import Path
from itertools import islice
import time as pytime
import csv, os
import numpy as np

lookahead = 5  # DeepPack3D의 탐색 길이
dp3d_model = 'rl'

TESTSET_DIR = Path("planning/data/Item_data/paper/testset")
RESULT_DIR  = Path(f"planning/Performance_comparison/EXP2_results/Deeppack3D_{dp3d_model}_without_const_l{lookahead}")
RESULT_DIR.mkdir(parents=True, exist_ok=True)



def main():
    json_files = sorted(TESTSET_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {TESTSET_DIR.resolve()}")

    # CSV 헤더는 한 번만
    csv_path = RESULT_DIR / "Deeppack3D_baseline_result.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow(["testdata", "packed_item_count", "SU"])

        for i, jf in enumerate(json_files, start=1):
            item_data = load_offline_data(str(jf))   # list[dict]
            print(f"[{jf.name}] 아이템 개수: {len(item_data)}")

            packer = Packer(
                problem="online",
                model="DeepPack3D",
                rotation_type=RotationType.BasicRotation,
                unfit_stop_setting=True,
                order_setting=False,
            )
            packer.robot_plag = False
            packer.packingModel.args.lookahead = lookahead
            packer.packingModel.args.method = dp3d_model
            packer.build_bin("experiment_RL")


            # 아이템을 packer에 추가
            for it in item_data:
                packer.addItem(Item(**it))

            start = pytime.time()
            packer.pack()
            # 500개마다 렌더 저장
            # if i % 10 == 0:
            render_dir = RESULT_DIR / f"{dp3d_model}_baseline_renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            packer.current_bin.render(
                save_path=str(render_dir),
                name=f'{jf.stem}_{packer.current_bin.size}_{packer.current_bin.getDensity():.2f}_final',
                show=False,
                save=True,
            )

            packer.log(f'packing end {pytime.time() - start:.4f}s, packed={packer.current_bin.size}')
            writer.writerow([jf.stem, packer.current_bin.size, f"{packer.current_bin.SU:.4f}"])
            print(f'>> [{jf.name}] packed_item_count={packer.current_bin.size}, SU={packer.current_bin.SU:.4f}')
            packer.items_list.clear()

    # ====== ⬇️ 여기서 CSV 전체를 읽어 평균/표준편차 계산 후 맨 아래에 추가 ======
    counts, sus = [], []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            # 요약행(__MEAN__/__STD__) 등은 건너뛰기
            if not row or len(row) < 3:
                continue
            try:
                c = float(row[1])
                s = float(row[2])
            except ValueError:
                # 숫자가 아니면(요약행 등) 스킵
                continue
            counts.append(c)
            sus.append(s)

    if counts and sus:
        counts = np.array(counts, dtype=float)
        sus    = np.array(sus, dtype=float)
        mean_count = counts.mean()
        std_count  = counts.std(ddof=1) if len(counts) > 1 else 0.0
        mean_su    = sus.mean()
        std_su     = sus.std(ddof=1) if len(sus) > 1 else 0.0

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([])  # 빈 줄로 본문-요약 구분
            w.writerow(["__MEAN__", f"{mean_count:.3f}", f"{mean_su:.4f}"])
            w.writerow(["__STD__",  f"{std_count:.3f}",  f"{std_su:.4f}"])



if __name__ == "__main__":
    main()