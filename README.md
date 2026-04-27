# PalletFit: Stability-Aware Online 3D Bin Packing
**via Transformer-Based Reinforcement Learning**

📌 Project page with videos: https://jinkyojb.github.io/PalletFit/

---

## Overview

Industrial palletizing increasingly requires **stable 3D bin packing** of items with diverse sizes.
However, many existing approaches restrict size variability and ignore **physical stability**, often leading to tilted or toppled placements.

**PalletFit** is an **online, stability-aware 3D bin packing framework** that unifies planning and execution through three stages:

- **Generation**: Edge-Projection (EDP) discretizes the continuous placement space into a compact candidate set.
- **Selection**: A Feature-Tokenizer Transformer-based RL agent jointly selects items and placements under strict feasibility constraints.
- **Execution**: A robot motion primitive, **Jam-Motion**, compensates for sensing and control errors via final contact alignment.

Extensive simulation and real-world experiments demonstrate dense and physically stable packing with **no observed stability violations** in real robot deployment.

---

## Method Highlights

- **Edge-Projection (EDP)**
  Converts continuous placement regions into a discrete, compact candidate set using lightweight geometric post-processing.

- **Transformer-based Reinforcement Learning**
  Jointly selects the next item and its placement using a Feature-Tokenizer Transformer with action masking to enforce feasibility constraints.

- **Jam-Motion Execution**
  A contact-based motion primitive that absorbs sensing and control errors before release, ensuring robust real-world placement.

---

## Experimental Results

- Significantly higher packing density compared to prior baselines
- No observed stability violations in real-world robot experiments
- Robust execution under sensing and control uncertainty

---

## Video Demos (×5 speed)

| Demo | Link |
|------|------|
| Demo 1 | https://youtu.be/enMSlwSKaNA |
| Demo 2 | https://youtu.be/q9RHRpqSUQI |
| Demo 3 | https://youtu.be/DRZbpDZeMbU |
| Demo 4 | https://youtu.be/TquNe8IssNA |
| Demo 5 | https://youtu.be/6wh6sItsSFQ |

Embedded videos and captions are available on the [project page](https://jinkyojb.github.io/PalletFit/).

---

## Installation

### Requirements

| Item | Version |
|------|---------|
| OS | Ubuntu 20.04 / 22.04 |
| Python | 3.10 |
| CUDA (optional) | 11.8 or 12.x |

### Setup

```bash
# 1. Clone
git clone https://github.com/JinkyoJB/PalletFit.git
cd PalletFit

# 2. Download pre-trained weights (Git LFS)
git lfs pull

# 3. Create environment
conda create -n palletfit python=3.10 -y
conda activate palletfit

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify
python -c "from planning.packer import Packer; print('OK')"
```

> **Headless server**: PyQt5 / PyOpenGL require display libraries.
> ```bash
> sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
> export QT_QPA_PLATFORM=offscreen
> ```

---

## Pre-trained Weights

Weights are stored via Git LFS in `planning/RL/PalletFit_RL/weight_files/`.

| File | Description |
|------|-------------|
| `ppo_ckpt_1265856_steps.zip` | Base checkpoint (default) |
| `fine_tunning_20260108.zip` | Fine-tuned checkpoint |

The active weight is set in `planning/packer.py` → `MODEL_REGISTRY["PalletFit_RL"]["init"]["weight_file_dir"]`.

---

## Quick Start

All commands are run from the **project root**.

### Demo — single online packing run

```bash
python demo.py
```

### Experiment 1 — Candidate-generation comparison

Compares EDP+POST, EDP, CP, EP, EMS on the custom item dataset.

```bash
python -c "from experiments.experiment1 import experiment1; experiment1()"
```

Results → `planning/experiment1_1265856_steps/` (CSV + PNG renders)

### Experiment 2 — Full evaluation on paper dataset

```bash
python -c "from experiments.experiment2 import experiment2; experiment2()"
```

Results → `planning/Performance_comparison/EXP2_results/`

### Palletizing demo

```bash
python -c "from experiments.palletizing import palletizing; palletizing()"
```

---

## RL Training

Training uses MaskablePPO (sb3-contrib) with 24 parallel environments.

```bash
# Full training loop (auto-restarts to avoid memory accumulation)
bash train.sh

# Single round
python planning/RL/PalletFit_RL/agent.py
```

`train.sh` runs `agent.py` in an infinite loop with automatic process cleanup between rounds. This is needed because 24+ parallel Gymnasium environments accumulate a gradual memory leak — each round trains for a fixed number of steps, saves a checkpoint, and exits cleanly before the loop relaunches it.

---

## Project Structure

```
PalletFit/
├── train.sh                        # training entry point (infinite loop w/ cleanup)
├── demo.py                         # demo / evaluation entry point
├── requirements.txt
│
├── experiments/                    # paper evaluation & demo scripts
│   ├── simulation.py               # main_simulation() — single online packing run
│   ├── experiment1.py              # EDP/CP/EP/EMS candidate-method comparison
│   ├── experiment2.py              # full evaluation on paper dataset
│   ├── palletizing.py              # industrial palletizing demo
│   ├── get_test.py                 # sanity test
│   └── results/                    # pre-computed result data & analysis scripts
│       ├── EXP1/                   # experiment1 CSV data per method
│       ├── EXP2_results/           # experiment2 per-baseline result CSVs & scripts
│       │   ├── FFD/                # FFD_baseline.py + result CSV
│       │   ├── PalletFit_RL/       # PalletFit_RL baseline.py + result CSVs
│       │   ├── Deeppack3D/         # DeepPack3D baseline.py + result CSVs
│       │   └── continuous_banila/  # continuous baseline log & cal_mean.py
│       ├── experiment1_1/          # step-level candidate count CSVs + graph.py
│       ├── csv_plot.py             # EDP_clean.csv visualisation
│       └── remove_triplicate_rows.py
│
├── planning/                       # core planning library
│   ├── packer.py                   # Packer class & MODEL_REGISTRY plugin system
│   ├── bin.py                      # Bin class (geometry, R-tree index)
│   ├── item.py                     # Item class (rotation, AABB)
│   ├── BinSpecsDict.py             # pallet dimension presets
│   ├── _stubs.py                   # hardware stubs (robot DSR, detection)
│   ├── heuristics/                 # FFD, ZoneFit, PalletFit, SafeFit, …
│   ├── RL/
│   │   └── PalletFit_RL/
│   │       ├── agent.py            # training logic (MaskablePPO)
│   │       ├── env.py              # Gymnasium environment
│   │       ├── custom_value_policy.py  # Feature-Tokenizer Transformer
│   │       ├── obs_builder.py      # observation engineering
│   │       ├── act_builder.py      # EDP action generation & placement
│   │       ├── reward_builder.py   # reward function
│   │       ├── rl_adapter.py       # inference adapter (Packer ↔ RL model)
│   │       ├── config.py           # PREVIEW_MAX, ACTION_MAX_CANDIDATES, …
│   │       └── weight_files/       # pre-trained checkpoints (Git LFS)
│   └── data/                       # item / bin JSON datasets
│       └── Item_data/
│           ├── paper/customset/    # experiment1 dataset
│           ├── paper/testset/      # RL overfit test
│           └── skt/                # SKT demo sets
│
├── utils/                          # geometry, pivot, painter helpers
│   ├── checkPivot.py
│   ├── pivot_generation.py
│   ├── get_value.py
│   └── painter/
│
└── docs/
    ├── 0_install.md
    └── troubleshooting_rl_distribution.md
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: planning` | Run from the project root: `cd PalletFit && python demo.py` |
| `FileNotFoundError` for weight file | Run `git lfs pull`; verify path in `MODEL_REGISTRY` |
| Qt / display error on server | `export QT_QPA_PLATFORM=offscreen` |
| CUDA OOM | Reduce `N_ENV` in `agent.py` or set `device="cpu"` |
| `OMP: Error #15` on macOS | `export KMP_DUPLICATE_LIB_OK=TRUE` |
| `ValueError: Simplex()` during inference | See [`docs/troubleshooting_rl_distribution.md`](docs/troubleshooting_rl_distribution.md) |
