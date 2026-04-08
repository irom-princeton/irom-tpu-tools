"""Unified TPU utilities for v4/v5/v6 with a single CLI.

This package provides:
- gcloud TPU VM wrappers with robust SSH flags and timeouts
- tmux helpers (launch, attach, ls, kill)
- admin helpers (kill JAX, clean tmp, nuke)
- list/delete helpers
- a unified watch-and-run launcher for v4/v5/v6
"""

from .config import TPUEnvConfig
from .ssh import SSHOptions
from .tpu import TPUManager

__all__ = [
    "SSHOptions",
    "TPUEnvConfig",
    "TPUManager",
]

PROJECT_NAME = "irom-tpu-tools"
