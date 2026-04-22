# planning/Performance_comparison/EXP_baseline.py
from planning.packer import Packer
from planning.item import Item, RotationType
from utils.util_functions import load_offline_data
from pathlib import Path
from itertools import islice
import time as pytime
import csv, os
import numpy as np


TESTSET_DIR = Path("planning/data/Item_data/paper/testset")
RESULT_DIR  = Path("planning/Performance_comparison/EXP2_results/FFD")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def FFD_baseline():
    json_files = sorted(TESTSET_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {TESTSET_DIR.resolve()}")

    # CSV 헤더는 한 번만
    csv_path = RESULT_DIR / "FFD_baseline_result.csv"
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
                model="FFD",
                rotation_type=RotationType.BasicRotation,
                unfit_stop_setting=True,
                order_setting=False,
            )
            packer.build_bin("experiment_RL")

            # 매 루프마다 1개만 미리보기 (FFD는 1개만 필요)
            failed = False
            while len(item_data) > 0 and not failed:
                preview_items = list(islice(item_data, 1))
                for it in preview_items:
                    packer.addItem(Item(**it))
                    packer.log(f"input item : [{it['width']}, {it['height']}, {it['depth']}]")

                    start = pytime.time()
                    packed_result = packer.pack()
                    packer.log(f'packing end {pytime.time() - start:.4f}s, packed={packer.current_bin.size}')

                    target_item = packer.get_target_item()

                    if target_item:
                        # 시각화용 플래그
                        target_item.options['is_attached'] = False
                        target_item.options['is_fixed'] = True
                        packer.items_list.clear()
                    
                        # 동일 name 한 개 제거
                        for k, it2 in enumerate(item_data):
                            if it2['name'] == target_item.name:
                                del item_data[k]
                                break
                    else:
                        print('------------------------적재 실패 또는 종료--------------------------')
                        failed = True
                        packer.items_list.clear()
                        
                        # 결과 기록 (파일별 1행)
                        writer.writerow([jf.stem, packer.current_bin.size, f"{packer.current_bin.SU:.4f}"])

                        # 500개마다 렌더 저장
                        if i % 10 == 0:
                            render_dir = RESULT_DIR / "FFD_baseline_renders"
                            render_dir.mkdir(parents=True, exist_ok=True)
                            packer.current_bin.render(
                                save_path=str(render_dir),
                                name=f'{jf.stem}_{packer.current_bin.size}_{packer.current_bin.getDensity():.2f}_final',
                                show=False,
                                save=True,
                            )

                        break  # ← for 탈출
                    
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
    FFD_baseline()
