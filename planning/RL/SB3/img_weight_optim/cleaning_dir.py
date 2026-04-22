import os
import re
import shutil

# 디렉토리 경로
ckpt_dir = "checkpoints"
snap_dir = "eval_snaps"

# 정규식: step과 fill값 추출
pattern = re.compile(r"step(\d+)_placed\d+_fill([\d.]+)\.png")

records = []  # (step, fill, ckpt_path, snap_path)

for subdir in os.listdir(snap_dir):
    snap_subdir = os.path.join(snap_dir, subdir)
    if not os.path.isdir(snap_subdir):
        continue
    for fname in os.listdir(snap_subdir):
        match = pattern.match(fname)
        if match:
            step = int(match.group(1))
            fill = float(match.group(2))
            ckpt_file = os.path.join(ckpt_dir, f"ckpt_{step:08d}.zip")
            snap_file = os.path.join(snap_subdir, fname)
            records.append((step, fill, ckpt_file, snap_file, snap_subdir))

# 성능이 가장 높은 것
best_fill = max(records, key=lambda x: x[1])
# step이 가장 큰 것 (가장 많이 학습된 것)
latest_step = max(records, key=lambda x: x[0])

keep_set = {best_fill[0], latest_step[0]}

print("남길 step:", keep_set)

# 삭제 실행
for step, fill, ckpt_file, snap_file, snap_subdir in records:
    if step not in keep_set:
        # 모델 파일 삭제
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
        # 스냅샷 디렉토리 삭제
        if os.path.exists(snap_subdir):
            shutil.rmtree(snap_subdir)

print("정리 완료!")
