"""
Wrapper to run Tabular Q-Learning training scaling experiment on Map 7x7 Hook.
Usage:
    python experiments/run_training_scaling.py --episodes 50000
    python experiments/run_training_scaling.py --episodes 100000
    python experiments/run_training_scaling.py --episodes 1000000
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_1m_episodes import run_1m

if __name__ == "__main__":
    run_1m()
