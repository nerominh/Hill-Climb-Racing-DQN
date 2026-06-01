"""Runtime helpers for keeping the RL code separate from the simulator code.

There are two reasonable ways this project can be run:

1. The environment package is already installed in the Python environment
2. The environment source is available locally in the sibling folder
   ``hillclimbracing``

We support both. That way the RL code stays outside the simulator package,
while still being convenient to run in a local repository checkout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_ROOT = REPO_ROOT / "hillclimbracing"


def ensure_simulator_on_path() -> None:
    """Make the environment package importable.

    We first check whether ``hill_racing_env`` is already installed and
    importable. If it is, we leave ``sys.path`` alone.

    If it is not installed, we fall back to the local repository layout by
    adding the sibling folder ``hillclimbracing`` to ``sys.path``.
    """

    # If the package is already installed, there is nothing to fix.
    if importlib.util.find_spec("hill_racing_env") is not None:
        return

    simulator_path = str(SIMULATOR_ROOT)
    if simulator_path not in sys.path:
        sys.path.insert(0, simulator_path)

    if importlib.util.find_spec("hill_racing_env") is None:
        raise ModuleNotFoundError(
            "Could not import 'hill_racing_env'. Install the simulator package "
            "or keep the local 'hillclimbracing' folder next to 'hcr_dqn'."
        )
