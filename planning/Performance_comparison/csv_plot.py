import pandas as pd
import matplotlib.pyplot as plt

# CSV 불러오기
df = pd.read_csv("planning/Performance_comparison/EDP_clean.csv")

# 확인
print(df.head())  # 상위 5행 확인

# 기본 시각화 (첫 번째 열을 X축, 나머지를 Y축)
plt.figure(figsize=(10, 5))
for col in df.columns[1:]:
    plt.plot(df[df.columns[0]], df[col], label=col)

plt.xlabel(df.columns[0])
plt.ylabel("Values")
plt.title("EDP_clean.csv Visualization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
