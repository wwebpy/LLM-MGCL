"""Symmetric normalisation and the container holding the three sparse adjacency matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


def build_norm_adj(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, device: torch.device
) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2} as a coalesced sparse COO tensor.

    Duplicate edges are summed by ``coalesce`` (e.g. mutual k-NN pairs appear twice and therefore
    receive twice the weight); the degree is computed on the same duplicated edge list.
    """
    row, col = edge_index[0].cpu(), edge_index[1].cpu()
    ew = edge_weight.float().cpu()

    deg = torch.zeros(num_nodes)
    deg.index_add_(0, row, ew)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    norm_w = d_inv_sqrt[row] * ew * d_inv_sqrt[col]
    return torch.sparse_coo_tensor(
        edge_index.cpu(), norm_w, (num_nodes, num_nodes), check_invariants=False
    ).coalesce().to(device)


def empty_adj(num_nodes: int, device: torch.device) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.zeros(2, 0, dtype=torch.long), torch.zeros(0), (num_nodes, num_nodes), check_invariants=False
    ).coalesce().to(device)


def item_edges_to_tensor(
    edges: Iterable[tuple[str, str, float]],
    item2idx: dict,
    weight_fn: Callable[[float], float] = float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map directed ``(business_a, business_b, value)`` edges to a bidirectional edge list in the
    local item space ``[0, n_items)``. Edges with unknown items are skipped."""
    src, dst, w = [], [], []
    for a, b, value in edges:
        if a not in item2idx or b not in item2idx:
            continue
        ia, ib = item2idx[a], item2idx[b]
        weight = weight_fn(value)
        src += [ia, ib]
        dst += [ib, ia]
        w += [weight, weight]
    return torch.tensor([src, dst], dtype=torch.long).reshape(2, -1), torch.tensor(w, dtype=torch.float)


@dataclass
class Graphs:
    """Normalised adjacencies: ``ui`` over users+items, ``sem`` and ``geo`` over items only."""

    ui: torch.Tensor
    sem: torch.Tensor
    geo: torch.Tensor
    n_users: int
    n_items: int

    def describe(self) -> str:
        return (f"UI {tuple(self.ui.shape)} nnz={self.ui._nnz():,} | "
                f"SEM {tuple(self.sem.shape)} nnz={self.sem._nnz():,} | "
                f"GEO {tuple(self.geo.shape)} nnz={self.geo._nnz():,}")


def build_graphs(edges: dict, n_users: int, n_items: int, device: torch.device) -> Graphs:
    """Build all three normalised adjacencies from the edge tensors stored by ``build_graphs.py``."""
    return Graphs(
        ui=build_norm_adj(edges["ui_edge_index"], edges["ui_edge_weight"], n_users + n_items, device),
        sem=build_norm_adj(edges["sem_edge_index"], edges["sem_edge_weight"], n_items, device),
        geo=build_norm_adj(edges["geo_edge_index"], edges["geo_edge_weight"], n_items, device),
        n_users=n_users,
        n_items=n_items,
    )
