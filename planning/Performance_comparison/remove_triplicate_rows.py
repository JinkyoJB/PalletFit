import pandas as pd

def remove_triplicate_rows(input_path: str, output_path: str):
    # CSV 읽기 (encoding은 상황에 맞게 utf-8, euc-kr 등으로 수정)
    df = pd.read_csv(input_path)

    # 중복 제거 (완전히 같은 행 기준)
    df_unique = df.drop_duplicates()

    # 결과 저장
    df_unique.to_csv(output_path, index=False)

    print(f"✅ 중복 제거 완료: {len(df) - len(df_unique)}개 행 제거됨")
    print(f"💾 저장 위치: {output_path}")

if __name__ == "__main__":
    input_file = "planning/Performance_comparison/EDP.csv"          # 원본 파일
    output_file = "planning/Performance_comparison/EDP_clean.csv"   # 결과 파일 이름
    remove_triplicate_rows(input_file, output_file)
