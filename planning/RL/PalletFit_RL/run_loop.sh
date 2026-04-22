#!/bin/bash

echo "Starting Infinite Training Loop with Auto-Cleanup..."

# =================================================================
# 1. [필수] CPU 과부하(Load Avg 700) 방지를 위한 쓰레드 제한
# =================================================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

count=1
while true; do
    echo "========================================================"
    echo "🚀 Round $count: Starting agent.py..."
    echo "========================================================"
    
    # 2. 파이썬 스크립트 실행
    # (agent.py 내부에서 학습 -> 저장 -> 종료 과정을 수행함)
    python planning/RL/PalletFit_RL/agent.py
    
    # 실행 종료 코드 확인 (비정상 종료 시 루프 멈춤)
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "❌ Error occurred (Exit Code: $exit_code). Stopping loop."
        break
    fi
    
    echo "💾 Round $count finished normally."
    
    # =================================================================
    # 3. [핵심] 좀비 프로세스 확인사살 (Cleanup)
    # =================================================================
    echo "🧹 Cleaning up lingering processes..."
    
    # 'agent.py'가 포함된 프로세스만 찾아서 강제 종료 (-f 옵션 사용)
    # 이렇게 해야 다른 python 프로그램은 살리고, 학습 코드만 확실히 죽입니다.
    pkill -9 -f "planning/RL/PalletFit_RL/agent.py"
    
    # 혹시 모르니 multiprocessing으로 생긴 자식 프로세스들도 정리
    # (주의: 이 부분은 현재 사용자(uon)의 모든 python 프로세스를 건드릴 수 있으므로
    #  위의 pkill -f 가 실패했을 때를 대비한 보험입니다. 불안하면 주석 처리하세요.)
    # pkill -u uon -9 python  <-- 이건 너무 강력해서 주석 처리함. 필요시 해제.

    echo "✅ Memory cleared. Cooldown for 5 seconds..."
    sleep 5
    ((count++))
done