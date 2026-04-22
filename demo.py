"""
PalletFit demo / evaluation entry point.

Usage (run from the project root):

    # Quick simulation with the default debug dataset
    python demo.py

    # Experiment 1 — candidate-generation method comparison
    python -c "from experiments.experiment1 import experiment1; experiment1()"

    # Experiment 2 — full evaluation on the paper dataset
    python -c "from experiments.experiment2 import experiment2; experiment2()"

    # Palletizing demo (area-buffered item feed)
    python -c "from experiments.palletizing import palletizing2; palletizing2()"
"""

from experiments.simulation import main_simulation
from experiments.palletizing import palletizing2
from experiments.experiment2 import experiment2

if __name__ == "__main__":
    problem_type = 'online'
    model = "PalletFit_RL"

    main_simulation(model, problem_type)
    # palletizing2(model=model)
    # experiment2()
