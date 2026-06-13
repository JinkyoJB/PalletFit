"""Centralised, machine-independent path resolution for PalletFit.

Open-source friendly: no personal absolute paths are hard-coded in the source
tree. Paths *inside* this repository are resolved relative to the repo root;
paths to *external* resources (baseline repositories, their datasets,
pretrained weights) are read from environment variables, which are loaded from
a local ``.env`` file. See ``.env.example`` for the full list and copy it to
``.env`` to configure your machine.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # python-dotenv is optional
    load_dotenv = None

# utils/env_paths.py  ->  parents[1] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")


def repo_path(*parts) -> Path:
    """Return an absolute path located inside this repository."""
    return REPO_ROOT.joinpath(*parts)


def env_path(var, default=None, required=False):
    """Resolve an external filesystem path from an environment variable.

    Configure it in your local ``.env`` (see ``.env.example``). Returns a
    ``Path`` (``~`` expanded) or ``None`` when unset and not required.
    """
    val = os.environ.get(var)
    if val:
        return Path(val).expanduser()
    if required:
        raise RuntimeError(
            f"Environment variable '{var}' is not set. Copy .env.example to "
            f".env and set '{var}' to the correct path on your machine.")
    return Path(default).expanduser() if default is not None else None


# --- External baseline repositories (optional; Experiment 2 reproduction) ---
PCT_REPO = env_path("PCT_REPO")
DRL_REPO = env_path("DRL_REPO")
GOPT_REPO = env_path("GOPT_REPO")
