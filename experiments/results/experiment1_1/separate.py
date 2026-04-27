import pandas as pd

# 1. 파일 불러오기
file_path = 'experiments/results/experiment1_1/EDP.csv'
df = pd.read_csv(file_path)

# 2. 에피소드 구분점 찾기
# 'step' 컬럼이 0인 행의 인덱스를 찾습니다. (새로운 에피소드의 시작)
start_indices = df.index[df['step'] == 0].tolist()
start_indices.append(len(df))  # 마지막 구간 처리를 위해 데이터프레임의 끝 인덱스 추가

# 3. 에피소드별로 데이터 쪼개기
episodes = []
for i in range(len(start_indices) - 1):
    start = start_indices[i]
    end = start_indices[i+1]
    episodes.append(df.iloc[start:end])

# 4. 순서에 따라 분리
# EDP_post: 첫 번째(인덱스 0), 세 번째(인덱스 2), 다섯 번째... -> 리스트 슬라이싱 [::2]
# EDP: 두 번째(인덱스 1), 네 번째(인덱스 3), 여섯 번째... -> 리스트 슬라이싱 [1::2]
df_post = pd.concat(episodes[::2], ignore_index=True)
df_edp = pd.concat(episodes[1::2], ignore_index=True)

# 5. 결과 파일 저장
df_post.to_csv('EDP_post.csv', index=False)
df_edp.to_csv('EDP_separated.csv', index=False)

print(f"분리 완료:")
print(f" - EDP_post 에피소드 수: {len(episodes[::2])}개")
print(f" - EDP 에피소드 수: {len(episodes[1::2])}개")