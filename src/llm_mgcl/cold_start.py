"""Item-popularity buckets for the cold-start analysis."""

from __future__ import annotations

from .data import Interactions


def assign_buckets(data: Interactions, bins: list[tuple[str, float, float]]) -> dict[int, str | None]:
    """Map every item index to the first bin ``(name, lo, hi)`` with ``lo <= count <= hi``.

    ``count`` is the number of training interactions of the item. Items outside all bins map to
    ``None`` and are ignored by :func:`llm_mgcl.evaluate.bucket_metrics`.
    """
    counts = data.item_train_counts()
    out: dict[int, str | None] = {}
    for item, idx in data.item2idx.items():
        c = int(counts.get(item, 0))
        out[idx] = next((name for name, lo, hi in bins if lo <= c <= hi), None)
    return out


def coarse_bins(cfg: dict) -> list[tuple[str, float, float]]:
    """Cold / warm / hot buckets from the config."""
    return [(name, float(lo), float(hi)) for name, (lo, hi) in cfg["cold_start"]["buckets"].items()]


def fine_bins(cfg: dict) -> dict[str, list[tuple[str, float, float]]]:
    """Fine bins per segment (``cold`` / ``warm`` / ``hot``) from the config."""
    return {seg: [(n, float(lo), float(hi)) for n, lo, hi in bins]
            for seg, bins in cfg["cold_start"]["fine_bins"].items()}


def items_per_bin(data: Interactions, bins: list[tuple[str, float, float]]) -> dict[str, int]:
    counts = data.item_train_counts()
    return {name: int(((counts >= lo) & (counts <= hi)).sum()) for name, lo, hi in bins}
