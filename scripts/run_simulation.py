"""Entry point: python scripts/run_simulation.py"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.digit_simulator import run_simulation

if __name__ == "__main__":
    run_simulation(n_samples=300)
