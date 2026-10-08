"""Neighbourhood-based collaborative filtering baselines (deterministic, no training)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ..data import Interactions


def item_user_matrix(train_df: pd.DataFrame, user2idx: dict, item2idx: dict) -> csr_matrix:
    """Sparse [n_items, n_users] matrix with star ratings as values."""
    df = train_df[train_df["user_id"].isin(user2idx) & train_df["business_id"].isin(item2idx)]
    rows = df["business_id"].map(item2idx).to_numpy()
    cols = df["user_id"].map(user2idx).to_numpy()
    vals = df["stars"].to_numpy(dtype=float)
    return csr_matrix((vals, (rows, cols)), shape=(len(item2idx), len(user2idx)))


def topk_cosine_neighbours(mat: csr_matrix, k: int = 50, chunk_size: int = 500, verbose=True):
    """Row-wise top-k cosine neighbours: ``{row: (neighbour_idx, similarity)}``."""
    n = mat.shape[0]
    topk = {}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = cosine_similarity(mat[start:end], mat)
        for local_i, global_i in enumerate(range(start, end)):
            row = np.asarray(sims[local_i]).ravel().copy()
            row[global_i] = -1
            idx = np.argpartition(row, -k)[-k:]
            topk[global_i] = (idx, row[idx])
        if verbose and start % 2000 == 0:
            print(f"  knn: {end}/{n}")
    return topk


class _KNNBase:
    def __init__(self, data: Interactions, k_neighbors: int = 50, chunk_size: int = 500, verbose=True):
        self.data = data
        self.n_items = data.n_items
        self.k = k_neighbors
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.item_user = item_user_matrix(data.train_df, data.user2idx, data.item2idx)

    def scores_np(self, u_raw) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def scorer(self, device: torch.device):
        """Same interface as the neural models: user indices [B] -> scores [B, n_items]."""
        idx2user = self.data.idx2user

        def score(users: torch.Tensor) -> torch.Tensor:
            mats = [self.scores_np(idx2user[u]) for u in users.tolist()]
            return torch.from_numpy(np.stack(mats).astype(np.float32)).to(device)

        return score


class ItemKNN(_KNNBase):
    """Score(i) = sum over the user's training items j of sim(i, j) * stars(u, j)  (sim > 0)."""

    def __init__(self, data: Interactions, **kwargs):
        super().__init__(data, **kwargs)
        self.star_lookup = data.train_df.set_index(["user_id", "business_id"])["stars"].to_dict()
        if self.verbose:
            print("ItemKNN: item-item similarities ...")
        self.sim_topk = topk_cosine_neighbours(self.item_user, self.k, self.chunk_size, self.verbose)

    def scores_np(self, u_raw):
        item2idx = self.data.item2idx
        scores = np.zeros(self.n_items)
        for i_raw in self.data.train_pos.get(u_raw, set()):
            if i_raw not in item2idx:
                continue
            star = float(self.star_lookup.get((u_raw, i_raw), 3.0))
            neighbours, sims = self.sim_topk[item2idx[i_raw]]
            for n_idx, sim in zip(neighbours, sims):
                if sim > 0:
                    scores[n_idx] += sim * star
        return scores


class UserKNN(_KNNBase):
    """Score(i) = sum over the top-k similar users v with sim > 0 that interacted with i of sim(u, v)."""

    def __init__(self, data: Interactions, **kwargs):
        super().__init__(data, **kwargs)
        if self.verbose:
            print("UserKNN: user-user similarities ...")
        self.sim_topk = topk_cosine_neighbours(self.item_user.T.tocsr(), self.k, self.chunk_size, self.verbose)

    def scores_np(self, u_raw):
        item2idx, idx2user = self.data.item2idx, self.data.idx2user
        scores = np.zeros(self.n_items)
        u_idx = self.data.user2idx[u_raw]
        if u_idx not in self.sim_topk:
            return scores
        neighbours, sims = self.sim_topk[u_idx]
        for n_idx, sim in zip(neighbours, sims):
            if sim <= 0:
                continue
            n_raw = idx2user.get(int(n_idx))
            if n_raw is None:
                continue
            for i_raw in self.data.train_pos.get(n_raw, set()):
                if i_raw in item2idx:
                    scores[item2idx[i_raw]] += sim
        return scores
