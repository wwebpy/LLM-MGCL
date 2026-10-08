"""Common interface for all trainable recommenders."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graphs import Graphs

ScoreFn = Callable[[torch.Tensor], torch.Tensor]  # user indices [B] -> scores [B, n_items]


def bpr_loss(u_e: torch.Tensor, p_e: torch.Tensor, n_e: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid((u_e * p_e).sum(1) - (u_e * n_e).sum(1)).mean()


def l2_reg(*embs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Sum of squared L2 norms of the given (ego) embeddings, divided by the batch size."""
    return sum(e.norm(2).pow(2) for e in embs) / batch_size


def info_nce(view_a: torch.Tensor, view_b: torch.Tensor, temp: float) -> torch.Tensor:
    """Symmetric InfoNCE between two aligned views; positives are on the diagonal."""
    a = F.normalize(view_a, dim=1)
    b = F.normalize(view_b, dim=1)
    logits = torch.matmul(a, b.T) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def lightgcn_propagate(adj: torch.Tensor, x: torch.Tensor, n_layers: int) -> torch.Tensor:
    """Parameter-free propagation; returns the mean over layer outputs 0..n_layers."""
    out = [x]
    for _ in range(n_layers):
        x = torch.sparse.mm(adj, x)
        out.append(x)
    return torch.stack(out, dim=0).mean(dim=0)


class Recommender(nn.Module):
    """Subclasses implement ``loss`` and either ``embeddings`` or ``scorer``."""

    def loss(self, users, pos, neg, graphs: Graphs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def embeddings(self, graphs: Graphs) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    @torch.no_grad()
    def scorer(self, graphs: Graphs) -> ScoreFn:
        """Score function used for evaluation (call after ``model.eval()``)."""
        user_emb, item_emb = self.embeddings(graphs)
        return lambda users: user_emb[users] @ item_emb.T
