"""Model registry."""

from __future__ import annotations

from .base import Recommender
from .baselines import GCMC, NGCF, SGL, LightGCN, MFBPR, NeuMF
from .knn import ItemKNN, UserKNN
from .llm_mgcl import LLMMGCL

TRAINABLE = {
    "llm_mgcl": LLMMGCL,
    "lightgcn": LightGCN,
    "ngcf": NGCF,
    "sgl": SGL,
    "gcmc": GCMC,
    "neumf": NeuMF,
    "mf_bpr": MFBPR,
}
KNN = {"itemknn": ItemKNN, "userknn": UserKNN}

DISPLAY_NAMES = {
    "llm_mgcl": "LLM-MGCL",
    "lightgcn": "LightGCN",
    "ngcf": "NGCF",
    "sgl": "SGL",
    "gcmc": "GC-MC",
    "neumf": "NeuMF",
    "mf_bpr": "MF-BPR",
    "itemknn": "ItemKNN",
    "userknn": "UserKNN",
}


def build_model(name: str, n_users: int, n_items: int, cfg: dict, variant: str | None = None) -> Recommender:
    """Instantiate a trainable model with the hyper-parameters from ``cfg['models'][name]``.

    ``variant`` (``full`` / ``wo_cl`` / ``wo_sem`` / ``wo_geo``) only applies to ``llm_mgcl``.
    """
    if name not in TRAINABLE:
        raise ValueError(f"unknown model '{name}', choose from {sorted(TRAINABLE)}")
    params = dict(cfg["models"][name])
    if name == "llm_mgcl":
        params.update(cfg["ablations"][variant or "full"])
    elif variant not in (None, "full"):
        raise ValueError("ablation variants are only defined for llm_mgcl")
    return TRAINABLE[name](n_users, n_items, **params)


__all__ = ["Recommender", "LLMMGCL", "LightGCN", "NGCF", "SGL", "GCMC", "NeuMF", "MFBPR",
           "ItemKNN", "UserKNN", "TRAINABLE", "KNN", "DISPLAY_NAMES", "build_model"]
