"""Uniform BPR triple sampler (user, positive item, negative item)."""

from __future__ import annotations

import numpy as np
import torch

from .data import Interactions


class BPRSampler:
    """Draws users uniformly, a positive uniformly from the user's training items, and a negative
    uniformly from all items (rejection sampling with up to ``max_tries`` attempts).

    Uses NumPy's global RNG, so ``set_seed`` makes it reproducible. Positives are kept in sorted
    order so the draws do not depend on Python's hash seed.
    """

    def __init__(self, data: Interactions, device: torch.device, max_tries: int = 20):
        self.device = device
        self.max_tries = max_tries
        self.n_items = data.n_items
        item2idx = data.item2idx

        # Users in sorted user_id order (same order as the groupby in `positive_sets`).
        users = [u for u in data.train_pos if u in data.user2idx]
        self.users = np.array([data.user2idx[u] for u in users])
        self.pos_list = {}
        self.pos_set = {}
        for u in users:
            idx = sorted(item2idx[i] for i in data.train_pos[u] if i in item2idx)
            self.pos_list[data.user2idx[u]] = np.array(idx)
            self.pos_set[data.user2idx[u]] = set(idx)

    def sample(self, batch_size: int = 2048):
        users, pos_items, neg_items = [], [], []
        chunk = batch_size * 3
        while len(users) < batch_size:
            for u in np.random.choice(self.users, size=chunk):
                if len(users) >= batch_size:
                    break
                pos_cands = self.pos_list[u]
                if len(pos_cands) == 0:
                    continue
                pos = np.random.choice(pos_cands)
                neg = None
                for _ in range(self.max_tries):
                    cand = np.random.randint(self.n_items)
                    if cand not in self.pos_set[u]:
                        neg = cand
                        break
                if neg is None:
                    continue
                users.append(u)
                pos_items.append(pos)
                neg_items.append(neg)

        as_tensor = lambda x: torch.tensor(np.asarray(x[:batch_size]), dtype=torch.long, device=self.device)
        return as_tensor(users), as_tensor(pos_items), as_tensor(neg_items)
