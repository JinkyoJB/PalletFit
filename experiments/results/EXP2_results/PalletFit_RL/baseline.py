# experiments/results/EXP2_results/PalletFit_RL/baseline.py

from planning.packer import Packer
from planning.item import Item, RotationType
from utils.util_functions import load_offline_data
from pathlib import Path
import time as pytime
import csv
import numpy as np

# 실험에 사용할 테스트셋 / 결과 폴더
TESTSET_DIR = Path("planning/data/Item_data/paper/testset")
RESULT_DIR  = Path("experiments/results/EXP2_results/PalletFit_RL_with_const_l1_20251201-194455_final_model")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    json_files = sorted(TESTSET_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {TESTSET_DIR.resolve()}")

    # CSV 헤더는 한 번만
    csv_path = RESULT_DIR / "PalletFit_RL_baseline_result.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow(["testdata", "packed_item_count", "SU"])

        for i, jf in enumerate(json_files, start=1):
            # ===== 1) 데이터 로드 =====
            item_data = load_offline_data(str(jf))   # list[dict]
            print(f"[{jf.name}] 아이템 개수: {len(item_data)}")

            # ===== 2) Packer 생성 (PalletFit_RL 사용) =====
            # COMPAT 상으로 ONLINE/ PALLETIZING에서 PalletFit_RL 허용됨
            # → MODEL_REGISTRY의 weight_file_dir을 따르는 rl_adapter 사용
            packer = Packer(
                problem="online",
                model="PalletFit_RL",
                rotation_type=RotationType.BasicRotation,
                unfit_stop_setting=True,
                order_setting=False,
            )
            packer.robot_plag = False
            packer.build_bin("experiment_RL")

            # ===== 3) 아이템 주입 =====
            for it in item_data:
                packer.addItem(Item(**it))

            # ===== 4) 패킹 실행 =====
            start = pytime.time()
            packer.pack()
            elapsed = pytime.time() - start

            # 10개마다 렌더 저장 (원하면 주기 조절)
            # if i % 10 == 0:
            render_dir = RESULT_DIR / "PalletFit_RL_baseline_renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            packer.current_bin.render(
                save_path=str(render_dir),
                name=f'{jf.stem}_{packer.current_bin.size}_{packer.current_bin.getDensity():.2f}_final',
                show=False,
                save=True,
            )

            # ===== 5) 로그 & CSV 기록 =====
            packer.log(
                f'PalletFit_RL packing end {elapsed:.4f}s, '
                f'packed={packer.current_bin.size}, SU={packer.current_bin.SU:.4f}'
            )
            writer.writerow([
                jf.stem,
                packer.current_bin.size,
                f"{packer.current_bin.SU:.4f}",
            ])
            print(
                f'>> [{jf.name}] packed_item_count={packer.current_bin.size}, '
                f'SU={packer.current_bin.SU:.4f}'
            )

            # 다음 테스트를 위해 아이템 리스트 초기화
            packer.items_list.clear()

    # ===== 6) CSV 전체 읽어서 평균/표준편차 요약행 추가 =====
    counts, sus = [], []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
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
            w.writerow([])  # 본문-요약 구분용 빈 줄
            w.writerow(["__MEAN__", f"{mean_count:.3f}", f"{mean_su:.4f}"])
            w.writerow(["__STD__",  f"{std_count:.3f}",  f"{std_su:.4f}"])


if __name__ == "__main__":
    main()
