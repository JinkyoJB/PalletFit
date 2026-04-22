# import pandas as pd

# # CSV 파일 읽기
# df = pd.read_csv("results.csv")

# # 각 파라미터 조합별로 메모리 사용량의 평균을 계산하여 요약 테이블 생성
# summary_table = df.groupby(['num_decay_epochs', 'replay_memory_size', 'batch_size', 'update_target_rate'])['memory_used_MB'].mean().reset_index()

# # 메모리 사용량을 기준으로 정렬 (옵션)
# summary_table = summary_table.sort_values(by='memory_used_MB', ascending=True)

# # 요약 테이블 출력
# print("Results Summary Table:")
# print(summary_table)

# # 필요에 따라 CSV로 저장
# summary_table.to_csv("summary_results.csv", index=False)
# print("Summary results saved to summary_results.csv")

# import seaborn as sns
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd

# # results.csv 파일 읽기
# df = pd.read_csv("results.csv")

# # 3D 산점도: num_decay_epochs, replay_memory_size, memory_used_MB를 축으로 사용
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(df['num_decay_epochs'], df['replay_memory_size'], df['memory_used_MB'],
#                 c=df['batch_size'], cmap='viridis', s=50)
# ax.set_xlabel('Num Decay Epochs')
# ax.set_ylabel('Replay Memory Size')
# ax.set_zlabel('Memory Used (MB)')
# plt.colorbar(sc, label='Batch Size')  # Batch Size를 색상으로 표시
# plt.title('3D Scatter Plot of Memory Usage by Parameter Combinations')
# plt.show()

# # Pairplot을 통해 각 파라미터와 memory_used_MB 간 상관 관계 시각화
# sns.pairplot(df, hue='memory_used_MB', palette="viridis", plot_kws={"s": 40})
# plt.suptitle('Pairplot of Parameter Combinations and Memory Usage', y=1.02)
# plt.show()

# # 히트맵: num_decay_epochs와 replay_memory_size에 따른 memory_used_MB의 평균
# heatmap_data = df.pivot_table(values='memory_used_MB', index='num_decay_epochs', columns='replay_memory_size', aggfunc='mean')
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Memory Usage (MB)'})
# plt.title('Memory Usage Heatmap by num_decay_epochs and replay_memory_size')
# plt.xlabel('Replay Memory Size')
# plt.ylabel('Num Decay Epochs')
# plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 결과 CSV 파일 읽기
df = pd.read_csv("results.csv")

# 고정 변수 값 설정 (여기서는 예시로 첫 번째 값들을 사용)
fixed_values = {
    "num_decay_epochs": 500,
    "replay_memory_size": 500,
    "batch_size": 16,
    "update_target_rate": 100
}

# 순회할 각 파라미터 리스트
param_lists = {
    "num_decay_epochs": [100, 200, 500, 1000],
    "replay_memory_size": [100, 200, 500, 1000],
    "batch_size": [4, 8, 16, 32],
    "update_target_rate": [50, 100, 200, 500]
}

# 한 파라미터씩 순회하며 메모리 사용량 시각화
for param, values in param_lists.items():
    # 고정 값에서 현재 파라미터만 변동
    filtered_df = df.copy()
    for key, value in fixed_values.items():
        if key != param:
            filtered_df = filtered_df[filtered_df[key] == value]
    
    # 결과 출력
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=param, y="memory_used_MB", data=filtered_df)
    plt.title(f"Memory Usage by {param} (with other parameters fixed)")
    plt.xlabel(param)
    plt.ylabel("Memory Used (MB)")
    plt.show()
