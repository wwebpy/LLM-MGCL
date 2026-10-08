"""Full-ranking top-K evaluation (Recall@K, NDCG@K), shared by every model.

Protocol (identical for all models):
* every user of the evaluation split with at least one known ground-truth item is evaluated;
* items from the user's training set are excluded from the ranking, and on the test split the
  user's validation items are excluded as well;
* Recall@K = hits / |ground truth|,  NDCG@K with IDCG over min(|ground truth|, K) positions.
"""

from __future__ import annotations

import math
from typing import Callable, Iterator

import numpy as np
import torch

from .data import Interactions

ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def _split_pos(data: Interactions, split: str) -> dict:
    return {"val": data.val_pos, "test": data.test_pos}[split]


@torch.no_grad()
def iter_rankings(
    score_fn: ScoreFn, data: Interactions, split: str, k: int, batch_size: int = 1024, device=None
) -> Iterator[tuple[str, list[int], list[int]]]:
    """Yield ``(user_id, top-k item indices, ground-truth item indices)`` per evaluated user."""
    eval_pos = _split_pos(data, split)
    item2idx, user2idx = data.item2idx, data.user2idx

    users, gts, masks = [], [], []
    for u_raw, items in eval_pos.items():
        if u_raw not in user2idx:
            continue
        gt = [item2idx[i] for i in items if i in item2idx]
        if not gt:
            continue
        seen = [item2idx[i] for i in data.train_pos.get(u_raw, set()) if i in item2idx]
        if split == "test":
            seen += [item2idx[i] for i in data.val_pos.get(u_raw, set()) if i in item2idx]
        users.append(u_raw)
        gts.append(gt)
        masks.append(seen)

    for start in range(0, len(users), batch_size):
        b_users = users[start:start + batch_size]
        u_idx = torch.tensor([user2idx[u] for u in b_users], dtype=torch.long, device=device)
        scores = score_fn(u_idx).clone()
        rows = [r for r, m in enumerate(masks[start:start + batch_size]) for _ in m]
        cols = [c for m in masks[start:start + batch_size] for c in m]
        if cols:
            scores[torch.tensor(rows, device=scores.device), torch.tensor(cols, device=scores.device)] = -1e9
        top = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        for r, u_raw in enumerate(b_users):
            yield u_raw, top[r].tolist(), gts[start + r]


def _recall_ndcg(topk: list[int], gt: set, k: int) -> tuple[float, float]:
    hits = [1 if idx in gt else 0 for idx in topk[:k]]
    recall = sum(hits) / len(gt)
    dcg = sum(h / math.log2(r + 2) for r, h in enumerate(hits))
    idcg = sum(1 / math.log2(r + 2) for r in range(min(len(gt), k)))
    return recall, (dcg / idcg if idcg > 0 else 0.0)


def per_user_metrics(score_fn: ScoreFn, data: Interactions, split: str = "test", ks=(10, 20),
                     batch_size: int = 1024, device=None) -> dict[str, tuple[float, ...]]:
    """``user_id -> (recall@k1, ndcg@k1, recall@k2, ndcg@k2, ...)``."""
    out = {}
    for u_raw, topk, gt in iter_rankings(score_fn, data, split, max(ks), batch_size, device):
        gt = set(gt)
        values = []
        for k in ks:
            values += list(_recall_ndcg(topk, gt, k))
        out[u_raw] = tuple(values)
    return out


def metric_names(ks=(10, 20)) -> list[str]:
    return [name for k in ks for name in (f"recall@{k}", f"ndcg@{k}")]


def summarize(per_user: dict, ks=(10, 20), users=None) -> dict[str, float]:
    """Mean over users (optionally restricted to ``users``)."""
    keys = users if users is not None else list(per_user)
    if not keys:
        return {m: 0.0 for m in metric_names(ks)}
    arr = np.array([per_user[u] for u in keys])
    return dict(zip(metric_names(ks), arr.mean(axis=0).tolist()))


def evaluate(score_fn: ScoreFn, data: Interactions, split: str = "test", ks=(10, 20),
             batch_size: int = 1024, device=None) -> dict[str, float]:
    return summarize(per_user_metrics(score_fn, data, split, ks, batch_size, device), ks)


def bucket_metrics(score_fn: ScoreFn, data: Interactions, item_bucket: dict, bucket_names: list[str],
                   split: str = "test", k: int = 20, batch_size: int = 1024, device=None) -> dict:
    """Recall@K / NDCG@K restricted to ground-truth items of each popularity bucket.

    The ranking is the normal full ranking; for each user the ground truth is split by
    ``item_bucket[item_idx]`` (items mapped to ``None`` or missing are ignored) and the metrics are
    computed per bucket. Users without ground truth in a bucket do not count for that bucket.
    """
    hits = {b: [] for b in bucket_names}
    ndcgs = {b: [] for b in bucket_names}
    for _, topk, gt in iter_rankings(score_fn, data, split, k, batch_size, device):
        rank = {idx: r for r, idx in enumerate(topk)}
        by_bucket = {b: [] for b in bucket_names}
        for g in gt:
            b = item_bucket.get(g)
            if b is not None:
                by_bucket[b].append(g)
        for b, gts in by_bucket.items():
            if not gts:
                continue
            hits[b].append(sum(1 for g in gts if g in rank) / len(gts))
            dcg = sum(1 / math.log2(rank[g] + 2) for g in gts if g in rank)
            idcg = sum(1 / math.log2(r + 2) for r in range(min(len(gts), k)))
            ndcgs[b].append(dcg / idcg if idcg > 0 else 0.0)
    return {
        b: ({"recall": float(np.mean(hits[b])), "ndcg": float(np.mean(ndcgs[b])),
             "n_users_with_gt": len(hits[b])} if hits[b] else None)
        for b in bucket_names
    }
