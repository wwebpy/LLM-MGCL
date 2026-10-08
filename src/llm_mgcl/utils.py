"""Configuration, seeding, device and checkpoint helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_name(model: str, variant: str | None = None) -> str:
    """Directory / file stem of a run, e.g. ``llm_mgcl`` or ``llm_mgcl_wo_cl``."""
    return model if variant in (None, "full") else f"{model}_{variant}"


def checkpoint_path(cfg: dict, model: str, seed: int, variant: str | None = None) -> Path:
    name = run_name(model, variant)
    return Path(cfg["paths"]["checkpoint_dir"]) / name / f"{name}_seed{seed}.pt"


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict | None:
    if not Path(path).exists():
        print(f"  missing: {path}")
        return None
    return torch.load(path, map_location=device, weights_only=False)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def mean_std(values) -> tuple[float, float]:
    """Mean and population std (``np.std``), as reported in the paper."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())
