"""
Run Best-of-N Baseline Sweep.

Wrapper script that executes the Best-of-N majority voting baseline
using the same dataset and evaluation protocol as the RLM sweep.
"""

from RLM.baselines.best_of_n import main

if __name__ == "__main__":
    main()
