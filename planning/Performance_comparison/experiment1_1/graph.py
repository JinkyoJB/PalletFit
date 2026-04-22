import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 파일 목록 및 라벨 정의
files = {
    'planning/Performance_comparison/experiment1_1/CP.csv': 'CP',
    'planning/Performance_comparison/experiment1_1/EDP_post.csv': 'EDP+post',
    'planning/Performance_comparison/experiment1_1/EDP_separated.csv': 'EDP',
    'planning/Performance_comparison/experiment1_1/EMS.csv': 'EMS',
    'planning/Performance_comparison/experiment1_1/EP.csv': 'EP'
}

# 그래프 설정
plt.figure(figsize=(10, 6))

# 스타일 설정
styles = {
    'CP': {'marker': 'o', 'linestyle': '-', 'color': 'tab:blue'},
    'EDP': {'marker': 's', 'linestyle': '--', 'color': 'tab:red'},
    'EDP+post': {'marker': 'x', 'linestyle': '-', 'color': 'tab:purple'},
    'EMS': {'marker': '^', 'linestyle': '-.', 'color': 'tab:green'},
    'EP': {'marker': 'v', 'linestyle': ':', 'color': 'tab:orange'}
}

print(f"{'Method':<15} | {'Max Packed Items':<20} | {'Overall Avg Candidates':<25}")
print("-" * 65)

# 2. 데이터 로드 및 시각화
for filename, label in files.items():
    if os.path.exists(filename):
        # CSV 읽기
        df = pd.read_csv(filename)
        
        # Step별로 그룹화하여 'feasible_candidates'의 평균 계산
        avg_candidates = df.groupby('step')['feasible_candidates'].mean()
        
        # [추가된 부분] 통계 출력 -----------------------------------------
        # 1. 최종 적재된 아이템 수 (X축의 최대값)
        max_packed_items = avg_candidates.index.max()
        
        # 2. 전체 평균 후보 수 (모든 스텝, 모든 에피소드의 평균)
        # 방법 A: 원래 데이터 전체의 평균 (가장 정확함)
        overall_avg = df['feasible_candidates'].mean()
        
        # 출력
        print(f"{label:<15} | {max_packed_items:<20} | {overall_avg:<25.2f}")
        # ---------------------------------------------------------------
        
        # 그래프 그리기
        style = styles.get(label, {})
        plt.plot(avg_candidates.index, avg_candidates.values, label=label, **style)

print("-" * 65)

# 3. 축 레이블 및 범례 설정
plt.xlabel('Number of Items Packed', fontsize=12)
plt.ylabel('Average Candidates', fontsize=12)
plt.title('Comparison of Candidate Generation Counts', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# 4. 저장 및 출력
plt.tight_layout()
output_path = 'planning/Performance_comparison/experiment1_1/candidates_comparison_plot.png'
# 디렉토리가 없으면 에러가 날 수 있으므로 확인 (선택 사항)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path, dpi=300)
plt.show()