"""LLM-MGCL: LightGCN with semantic and geographic item views plus contrastive alignment."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..graphs import Graphs
from .base import Recommender, bpr_loss, info_nce, l2_reg, lightgcn_propagate


class LLMMGCL(Recommender):
    """Three parallel LightGCN propagation paths sharing the same ego embeddings.

    * G_UI  - collaborative signal on the user-item graph (users and items)
    * G_SEM - item-item graph from LLM photo-summary embeddings
    * G_GEO - item-item graph from Haversine distances

    Item representation: additive fusion ``e_cf + e_sem + e_geo`` of the active views.
    Loss: BPR + L2 + ``lam`` * InfoNCE(e_cf, e_view) averaged over the active views.

    ``use_sem`` / ``use_geo`` / ``use_cl`` switch the components off for the ablation study; the
    flags apply consistently to training and evaluation.
    """

    def __init__(self, n_users, n_items, emb_dim=64, n_layers_ui=3, n_layers_sim=2, n_layers_geo=2,
                 temp=0.2, reg=1e-3, lam=0.05, use_sem=True, use_geo=True, use_cl=True):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.temp = temp
        self.n_layers_ui = n_layers_ui
        self.n_layers_sim = n_layers_sim
        self.n_layers_geo = n_layers_geo
        self.reg = reg
        self.lam = lam if use_cl else 0.0
        self.use_sem = use_sem
        self.use_geo = use_geo

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, ui_adj, sim_adj, geo_adj):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_cf = lightgcn_propagate(ui_adj, ego, self.n_layers_ui)
        user_cf, item_cf = all_cf[: self.n_users], all_cf[self.n_users:]
        item_sem = lightgcn_propagate(sim_adj, self.item_emb.weight, self.n_layers_sim)
        item_geo = lightgcn_propagate(geo_adj, self.item_emb.weight, self.n_layers_geo)
        return user_cf, item_cf, item_sem, item_geo

    def fuse(self, item_cf, item_sem, item_geo):
        item = item_cf.clone()
        if self.use_sem:
            item = item + item_sem
        if self.use_geo:
            item = item + item_geo
        return item

    def loss(self, users, pos, neg, graphs: Graphs):
        user_cf, item_cf, item_sem, item_geo = self.forward(graphs.ui, graphs.sem, graphs.geo)
        item_final = self.fuse(item_cf, item_sem, item_geo)

        bpr = bpr_loss(user_cf[users], item_final[pos], item_final[neg])
        reg = l2_reg(self.user_emb(users), self.item_emb(pos), self.item_emb(neg), batch_size=users.size(0))

        cl_terms = []
        if self.use_sem:
            cl_terms.append(info_nce(item_cf[pos], item_sem[pos], self.temp))
        if self.use_geo:
            cl_terms.append(info_nce(item_cf[pos], item_geo[pos], self.temp))
        cl = sum(cl_terms) / len(cl_terms) if cl_terms else 0.0

        return bpr + self.reg * reg + self.lam * cl

    def embeddings(self, graphs: Graphs):
        user_cf, item_cf, item_sem, item_geo = self.forward(graphs.ui, graphs.sem, graphs.geo)
        return user_cf, self.fuse(item_cf, item_sem, item_geo)

    def view_embeddings(self, graphs: Graphs) -> dict[str, torch.Tensor]:
        """Per-view item embeddings (for analysis)."""
        user_cf, item_cf, item_sem, item_geo = self.forward(graphs.ui, graphs.sem, graphs.geo)
        return {"user": user_cf, "cf": item_cf, "sem": item_sem, "geo": item_geo}
