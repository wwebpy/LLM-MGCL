"""User-item interaction graph G_UI."""

from __future__ import annotations

import pandas as pd
import torch


def build_ui_edges(
    train_df: pd.DataFrame, user2idx: dict, item2idx: dict, star_weight_min: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected user-item edges in the joint node space ``[0, n_users + n_items)``.

    Edge weights are the star ratings mapped from [1, 5] to [0, 1] and clamped from below at
    ``star_weight_min`` so that 1-star reviews do not vanish from the graph.

    Returns ``edge_index`` of shape ``[2, 2 * |train|]`` and the matching ``edge_weight``.
    """
    n_users = len(user2idx)
    u = torch.tensor(train_df["user_id"].map(user2idx).values, dtype=torch.long)
    i = torch.tensor(train_df["business_id"].map(item2idx).values, dtype=torch.long) + n_users

    edge_index = torch.stack([torch.cat([u, i]), torch.cat([i, u])], dim=0)

    stars = torch.tensor(train_df["stars"].values, dtype=torch.float)
    star_w = torch.clamp((stars - 1.0) / 4.0, min=star_weight_min)
    edge_weight = torch.cat([star_w, star_w], dim=0)
    return edge_index, edge_weight
