"""Loading the preprocessed dataset (written by ``scripts/build_graphs.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .data import Interactions
from .graphs import Graphs, build_graphs

SPLIT_DTYPES = {"user_id": str, "business_id": str, "stars": float}


@dataclass
class Dataset:
    data: Interactions
    graphs: Graphs


def processed_files(processed_dir: str | Path) -> dict[str, Path]:
    d = Path(processed_dir)
    return {
        "train": d / "train.csv",
        "val": d / "val.csv",
        "test": d / "test.csv",
        "edges": d / "edges.pt",
        "embeddings": d / "sem_embeddings.npz",
        "item_texts": d / "item_texts.csv",
    }


def load_dataset(cfg: dict, device: torch.device, verbose: bool = True) -> Dataset:
    files = processed_files(cfg["paths"]["processed_dir"])
    if not files["edges"].exists():
        raise FileNotFoundError(
            f"{files['edges']} not found - run `python scripts/build_graphs.py` first."
        )
    data = Interactions(
        pd.read_csv(files["train"], dtype=SPLIT_DTYPES, keep_default_na=False),
        pd.read_csv(files["val"], dtype=SPLIT_DTYPES, keep_default_na=False),
        pd.read_csv(files["test"], dtype=SPLIT_DTYPES, keep_default_na=False),
    )
    edges = torch.load(files["edges"], weights_only=False)
    # The integer ids must match the ones the edges were built with.
    assert list(data.user2idx) == edges["users"], "user order differs from edges.pt"
    assert list(data.item2idx) == edges["items"], "item order differs from edges.pt"

    graphs = build_graphs(edges, data.n_users, data.n_items, device)
    if verbose:
        print(f"Users {data.n_users:,} | Items {data.n_items:,} | Train {len(data.train_df):,} | "
              f"Val {len(data.val_df):,} | Test {len(data.test_df):,}")
        print(graphs.describe())
    return Dataset(data, graphs)


def load_trained_model(name: str, seed: int, cfg: dict, data: Interactions, device: torch.device,
                       variant: str | None = None):
    """Rebuild a trained model from ``checkpoints/<run>/<run>_seed<seed>.pt`` (``None`` if missing)."""
    from .models import build_model
    from .utils import checkpoint_path, load_checkpoint

    ckpt = load_checkpoint(checkpoint_path(cfg, name, seed, variant), device)
    if ckpt is None:
        return None
    model = build_model(name, data.n_users, data.n_items, cfg, variant).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
