def calculate_averages(filename):
    """
    텍스트 파일을 읽어 packed와 SU의 평균을 계산하는 함수
    
    Args:
        filename (str): 데이터 파일의 경로
        
    Returns:
        tuple: (평균 packed, 평균 SU)
    """
    packed_sum = 0
    su_sum = 0
    count = 0

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 빈 줄이나 구분선('---')은 건너뜀
                if not line.strip() or line.startswith('-'):
                    continue
                
                # 쉼표로 분리하고 각 부분에서 값 추출
                parts = line.strip().split(', ')
                packed_val = 0
                su_val = 0.0
                
                for part in parts:
                    if part.startswith('packed='):
                        packed_val = int(part.split('=')[1])
                    elif part.startswith('SU='):
                        su_val = float(part.split('=')[1])
                
                packed_sum += packed_val
                su_sum += su_val
                count += 1
        
        if count == 0:
            return 0, 0.0

        avg_packed = packed_sum / count
        avg_su = su_sum / count
        std_su = (su_sum / count) ** 0.5  # 표준편차 계산
        std_packed = (packed_sum / count) ** 0.5
        
        return avg_packed, avg_su, std_packed, std_su

    except FileNotFoundError:
        print(f"오류: '{filename}' 파일을 찾을 수 없습니다.")
        return None, None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None, None

# --- 사용 예시 ---
# 1. 위 텍스트 내용을 'data.txt'라는 파일로 저장했다고 가정합니다.
# 2. 함수를 호출하여 결과를 확인합니다.

filename = 'planning/Performance_comparison/EXP2_results/continuous_banila/result_log_banila.txt'  # 실제 파일 경로로 변경하세요
avg_packed, avg_su, std_packed, std_su = calculate_averages(filename)

if avg_packed is not None:
    print(f"분석된 데이터 개수: {100}개 (추정)") # 실제 count 변수 활용 가능
    print(f"평균 packed: {avg_packed:.4f}")
    print(f"표준편차 packed: {std_packed:.4f}")
    print(f"평균 SU: {avg_su:.4f}")
    print(f"표준편차 SU: {std_su:.4f}")