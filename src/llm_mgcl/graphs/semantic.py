"""Semantic item-item graph G_SEM built from LLM-generated photo summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import haversine_to_all


def build_item_texts(
    photo_summary: pd.DataFrame, train_items: set, text_fields=("summary", "keywords")
) -> pd.DataFrame:
    """Keep training items only (avoids leakage) and concatenate the text fields.

    Returns a frame with ``business_id`` and ``text_combined``; rows with empty text are dropped.
    """
    ps = photo_summary[photo_summary["business_id"].isin(train_items)].copy()
    text = ps[text_fields[0]].fillna("")
    for col in text_fields[1:]:
        text = text + " " + ps[col].fillna("")
    ps["text_combined"] = text.str.strip()
    ps = ps[ps["text_combined"].str.len() > 0].reset_index(drop=True)
    return ps


def encode_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """L2-normalised Sentence-Transformer embeddings (so that dot product = cosine similarity)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def build_semantic_edges(
    embeddings: np.ndarray,
    business_ids: list[str],
    business: pd.DataFrame,
    k: int = 10,
    max_dist_km: float = 50.0,
    batch_size: int = 1024,
    verbose: bool = True,
) -> list[tuple[str, str, float]]:
    """For every item, link the ``k`` most similar items (cosine > 0) within ``max_dist_km``.

    The distance filter restricts semantic neighbours to the same area (same threshold as G_GEO).
    Items without coordinates get no semantic neighbours.

    Returns directed edges ``(business_a, business_b, cosine_similarity)``, ordered by source item
    and then by descending similarity.
    """
    coords = business.drop_duplicates("business_id").set_index("business_id")[["latitude", "longitude"]]
    lat = coords["latitude"].reindex(business_ids).to_numpy(dtype=float)
    lon = coords["longitude"].reindex(business_ids).to_numpy(dtype=float)
    has_coord = np.isin(np.asarray(business_ids), coords.index.to_numpy())

    emb = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms == 0, 1.0, norms)

    n = len(business_ids)
    edges: list[tuple[str, str, float]] = []
    degrees = np.zeros(n, dtype=int)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = emb[start:end] @ emb.T                                  # cosine similarity
        sims[np.arange(end - start), np.arange(start, end)] = -1.0     # no self-loops
        dist = haversine_to_all(lat[start:end], lon[start:end], lat, lon)
        with np.errstate(invalid="ignore"):
            valid = (sims > 0) & (dist <= max_dist_km)                 # NaN / missing -> False
        masked = np.where(valid, sims, -np.inf)

        for row in range(end - start):
            i = start + row
            if not has_coord[i]:
                continue
            cand = np.argpartition(-masked[row], min(k, n - 1))[: k + 1]
            cand = cand[np.argsort(-masked[row, cand])]
            kept = 0
            for j in cand:
                if not np.isfinite(masked[row, j]):
                    break
                edges.append((business_ids[i], business_ids[j], float(sims[row, j])))
                kept += 1
                if kept >= k:
                    break
            degrees[i] = kept

    if verbose:
        print(f"  sem: {len(edges):,} directed edges | mean degree {degrees.mean():.2f} | "
              f"items without coordinates {int((~has_coord).sum())} | "
              f"items with < {k} neighbours {int((degrees < k).sum())} | "
              f"isolated {int((degrees == 0).sum())}")
    return edges
