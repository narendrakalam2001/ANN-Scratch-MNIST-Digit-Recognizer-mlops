"""Entry point: python scripts/run_dashboard.py"""

import subprocess
import sys
import os

if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            os.path.join(root, "monitoring", "monitoring_dashboard.py"),
        ]
    )
