# PalletFit: Stability-Aware Online 3D Bin Packing  
**via Transformer-Based Reinforcement Learning**

📌 Project page with videos:  
👉 https://jinkyojb.github.io/PalletFit/

---

## 📄 Overview
Industrial palletizing increasingly requires **stable 3D bin packing** of items with diverse sizes.  
However, many existing approaches restrict size variability and ignore **physical stability**, often leading to tilted or toppled placements.

**PalletFit** is an **online, stability-aware 3D bin packing framework** that unifies planning and execution through three stages:

- **Generation**: Edge-Projection (EDP) discretizes the continuous placement space into a compact candidate set.
- **Selection**: A Feature-Tokenizer Transformer-based RL agent jointly selects items and placements under strict feasibility constraints.
- **Execution**: A robot motion primitive, **Jam-Motion**, compensates for sensing and control errors via final contact alignment.

Extensive simulation and real-world experiments demonstrate dense and physically stable packing with **no observed stability violations** in real robot deployment.

---

## 🎥 Video Demos (×5 speed)
All videos are shown at **×5 speed** unless otherwise stated.

| Demo | Link |
|----|----|
| Demo 1 | https://youtu.be/enMSlwSKaNA |
| Demo 2 | https://youtu.be/q9RHRpqSUQI |
| Demo 3 | https://youtu.be/DRZbpDZeMbU |
| Demo 4 | https://youtu.be/TquNe8IssNA |
| Demo 5 | https://youtu.be/6wh6sItsSFQ |

▶️ **Embedded videos and captions are available on the project page:**  
https://jinkyojb.github.io/PalletFit/

---

## 🧠 Method Highlights
- **Edge-Projection (EDP)**  
  Converts continuous placement regions into a discrete, compact candidate set using lightweight geometric post-processing.

- **Transformer-based Reinforcement Learning**  
  Jointly selects the next item and its placement using a Feature-Tokenizer Transformer with action masking to enforce feasibility constraints.

- **Jam-Motion Execution**  
  A contact-based motion primitive that absorbs sensing and control errors before release, ensuring robust real-world placement.

---

## 🧪 Experimental Results
- Significantly higher packing density compared to prior baselines
- No observed stability violations in real-world robot experiments
- Robust execution under sensing and control uncertainty

(See the project page for videos and qualitative results.)

---

## 🔁 Reproducibility
Reproducibility instructions will be provided upon code release.

```text
# Coming soon
# Installation, training, and evaluation instructions
