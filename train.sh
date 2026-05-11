#!/bin/bash
# Training entry point for PalletFit-RL (MaskablePPO).
#
# Runs agent.py in an infinite loop with automatic process cleanup between
# rounds. This pattern is necessary because multiprocessing environments
# (24+ parallel envs) accumulate a small memory leak over time that would
# eventually stall training. Each round trains for a fixed number of steps,
# saves a checkpoint, then exits cleanly; this script relaunches it.

echo "Starting PalletFit-RL Training Loop..."

# ─── PYTHONPATH: 프로젝트 root를 import 경로에 포함 (planning.* import용) ────
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH}"

# ─── PyTorch CUDA: fragmentation 완화 (큰 minibatch alloc 시 안전망) ─────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── CPU thread limits (prevents Load Avg 700+ on multi-core machines) ───────
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

    python planning/RL/PalletFit_RL/agent.py "$@"

    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "❌ Error occurred (Exit Code: $exit_code). Stopping loop."
        break
    fi

    echo "💾 Round $count finished normally."

    # ─── Kill any lingering subprocesses from the previous round ─────────────
    echo "🧹 Cleaning up lingering processes..."
    pkill -9 -f "planning/RL/PalletFit_RL/agent.py"

    echo "✅ Memory cleared. Cooldown for 5 seconds..."
    sleep 5
    ((count++))
done
