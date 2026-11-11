"""Config loading, seeding and small shared helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"
VAL_SEED_OFFSET = 10_000  # validation batches and pairs are drawn with seed + this, so they never repeat the training draws


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read the config: the given file, else the file named by PERTACT_CONFIG, else config.yaml in the repository."""
    with open(path or os.environ.get("PERTACT_CONFIG") or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def path(cfg: dict, key: str) -> Path:
    """Resolve cfg['paths'][key] relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and, if it is installed, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    """A logger with the format every module uses."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    return logging.getLogger(name)


def write_json(obj, p: Path) -> None:
    """Write JSON, creating the folder if needed. NumPy numbers become plain ones."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def read_json(p: Path):
    """Read a JSON file."""
    with open(p) as f:
        return json.load(f)
