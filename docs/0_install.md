# PalletFit — Installation & Quick-Start Guide

## Overview

This repository contains the **planning module** of PalletFit, a stability-aware online 3D bin-packing framework. It includes:

| Component | Location |
|---|---|
| Packing algorithms (heuristics) | `planning/heuristics/` |
| Reinforcement-learning agent (MaskablePPO) | `planning/RL/PalletFit_RL/` |
| Utility functions (geometry, visualisation) | `utils/` |
| Pre-trained weight files | `planning/RL/PalletFit_RL/weight_files/` |
| Evaluation & demo scripts | `experiments/` |

> **Note**: Robot control (`dsrpy`) and camera perception (`detection`) modules from the full system are not included. Stub implementations are provided so the planning code runs without any hardware.

---

## 1. Requirements

| Item | Version |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| Python | 3.10 |
| CUDA (optional, for GPU) | 11.8 or 12.x |

---

## 2. Environment Setup

### 2-1. Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Or with conda:

```bash
conda create -n palletfit python=3.10 -y
conda activate palletfit
```

### 2-2. Install dependencies

```bash
pip install -r requirements.txt
```

> PyQt5 / PyOpenGL require display libraries on headless servers. If you run without a display, install:
> ```bash
> sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
> ```
> and set `export QT_QPA_PLATFORM=offscreen` before running.

### 2-3. Verify installation

```bash
python -c "from planning.packer import Packer; print('OK')"
```

---

## 3. Pre-trained Weights

Two weight files are included in `planning/RL/PalletFit_RL/weight_files/`:

| File | Description |
|---|---|
| `ppo_ckpt_1265856_steps.zip` | Base checkpoint (used by default) |
| `fine_tunning_20260108.zip` | Fine-tuned checkpoint |

The active weight is configured in `planning/packer.py` → `MODEL_REGISTRY["PalletFit_RL"]["init"]["weight_file_dir"]`.

> **Git LFS**: Weight files are stored with Git LFS. After cloning, run `git lfs pull` to download the actual binaries.

---

## 4. Running Experiments

All scripts are run from the **project root** (`PalletFit/`).

### 4-1. Quick simulation (single run)

```bash
python demo.py
```

This calls `main_simulation()` with the default debug dataset and renders the result.

### 4-2. Experiment 1 — Candidate-generation comparison

Compares five placement-candidate methods (EDP+POST, EDP, CP, EP, EMS) on the custom item dataset.

```bash
# Prepare dataset first (see Section 5)
python -c "from experiments.experiment1 import experiment1; experiment1()"
```

Results are saved to `planning/experiment1_1265856_steps/`:
- `experiment1_results.csv` — per-file packed count, SU, and time
- PNG renders of each packed bin

### 4-3. Experiment 2 — Full evaluation on paper dataset

```bash
python -c "from experiments.experiment2 import experiment2; experiment2()"
```

Results are saved under `planning/Performance_comparison/EXP2_results/`.

### 4-4. Palletizing demo (area-buffered item feed)

```bash
python -c "from experiments.palletizing import palletizing; palletizing()"
```

---

## 5. Dataset

### Using included data

Sample item JSON files are located in:

```
planning/data/Item_data/
├── debug/          # small debug sets
├── paper/
│   ├── customset/  # used by experiment1, experiment1_1
│   └── testset/    # used by RL overfit test
└── skt/            # SKT demo sets
```

### Generating new datasets

```bash
python planning/data/items_generator.py
```

---

## 6. RL Training

Training uses MaskablePPO from `sb3-contrib`. To resume or start a new training run:

```bash
bash train.sh
```

This script runs `planning/RL/PalletFit_RL/agent.py` in an infinite loop with automatic process cleanup between rounds. Each round trains for a fixed number of steps, saves a checkpoint, and exits; the loop relaunches it to work around a gradual memory leak from 24+ parallel environments.

To run a single training round:

```bash
python planning/RL/PalletFit_RL/agent.py
```

Training checkpoints are saved to the path configured inside `agent.py`.

---

## 7. Visualisation

Results can be rendered with Open3D or Matplotlib. The `render` flag in most functions controls whether a window opens during execution.

```python
packer.current_bin.render(show=True, save=False)
```

On a headless server, set `show=False` and `save=True` to write PNG files instead.

---

## 8. Project Structure

```
PalletFit/
├── train.sh                    # training entry point (infinite loop w/ cleanup)
├── demo.py                     # quick demo / evaluation entry point
├── requirements.txt
│
├── experiments/                # paper evaluation & demo scripts
│   ├── simulation.py           # main_simulation() — single online packing run
│   ├── experiment1.py          # EDP/CP/EP/EMS candidate-method comparison
│   ├── experiment2.py          # full evaluation on paper dataset
│   ├── palletizing.py          # industrial palletizing demo
│   └── get_test.py             # quick sanity test
│
├── planning/                   # core planning library
│   ├── packer.py               # main Packer class & MODEL_REGISTRY
│   ├── bin.py                  # Bin class
│   ├── item.py                 # Item class
│   ├── BinSpecsDict.py         # pallet dimension presets
│   ├── _stubs.py               # hardware stubs (robot, detection)
│   ├── heuristics/             # FFD, ZoneFit, PalletFit, SafeFit, …
│   ├── RL/
│   │   └── PalletFit_RL/       # MaskablePPO agent + environment
│   │       ├── agent.py        # training logic
│   │       ├── env.py          # Gymnasium environment
│   │       ├── custom_value_policy.py  # Feature-Tokenizer Transformer
│   │       ├── obs_builder.py  # observation engineering
│   │       ├── act_builder.py  # action generation & placement
│   │       ├── reward_builder.py
│   │       ├── rl_adapter.py   # inference adapter used by Packer
│   │       ├── config.py       # PREVIEW_MAX, ACTION_MAX_CANDIDATES, …
│   │       └── weight_files/   # pre-trained checkpoints (Git LFS)
│   └── data/                   # item / bin JSON datasets
│
├── utils/                      # geometry, pivot, painter helpers
└── docs/
    ├── 0_install.md            # this file
    └── troubleshooting_rl_distribution.md
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: planning` | Run from the project root: `cd PalletFit && python demo.py` |
| `FileNotFoundError` for weight file | Run `git lfs pull`; check `MODEL_REGISTRY["PalletFit_RL"]["init"]["weight_file_dir"]` in `planning/packer.py` |
| Qt / display error on server | `export QT_QPA_PLATFORM=offscreen` |
| CUDA OOM | Reduce `N_ENV` in `agent.py` or switch to CPU: `device="cpu"` in `MODEL_REGISTRY` |
| `OMP: Error #15` on macOS | `export KMP_DUPLICATE_LIB_OK=TRUE` |
| `ValueError: Simplex()` during inference | See `docs/troubleshooting_rl_distribution.md` |
