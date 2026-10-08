"""Trainable baselines: LightGCN, NGCF, SGL, GC-MC, NeuMF, MF-BPR.

All baselines use the same BPR training loop, sampler and evaluation as LLM-MGCL. Parameter names
are kept identical to the original experiment code so that existing checkpoints can be loaded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graphs import Graphs
from .base import Recommender, bpr_loss, info_nce, l2_reg, lightgcn_propagate


class LightGCN(Recommender):
    """He et al., SIGIR 2020."""

    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3, reg=1e-3):
        super().__init__()
        self.n_users, self.n_items, self.n_layers, self.reg = n_users, n_items, n_layers, reg
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, ui_adj):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_emb = lightgcn_propagate(ui_adj, ego, self.n_layers)
        return all_emb[: self.n_users], all_emb[self.n_users:]

    def embeddings(self, graphs: Graphs):
        return self.forward(graphs.ui)

    def loss(self, users, pos, neg, graphs: Graphs):
        user_emb, item_emb = self.forward(graphs.ui)
        bpr = bpr_loss(user_emb[users], item_emb[pos], item_emb[neg])
        reg = l2_reg(self.user_emb(users), self.item_emb(pos), self.item_emb(neg), batch_size=users.size(0))
        return bpr + self.reg * reg


class NGCF(Recommender):
    """Wang et al., SIGIR 2019."""

    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3, dropout=0.1, reg=1e-3):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.dropout, self.reg = dropout, reg
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.W1 = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)])

    def forward(self, ui_adj):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embs, x = [ego], ego
        for layer in range(self.n_layers):
            neighbor = torch.sparse.mm(ui_adj, x)
            out = F.leaky_relu(self.W1[layer](neighbor) + self.W2[layer](x * neighbor))
            out = F.dropout(out, p=self.dropout, training=self.training)
            out = F.normalize(out, dim=1)
            all_embs.append(out)
            x = out
        all_embs = torch.stack(all_embs, dim=0).mean(dim=0)
        return all_embs[: self.n_users], all_embs[self.n_users:]

    def embeddings(self, graphs: Graphs):
        return self.forward(graphs.ui)

    def loss(self, users, pos, neg, graphs: Graphs):
        user_emb, item_emb = self.forward(graphs.ui)
        bpr = bpr_loss(user_emb[users], item_emb[pos], item_emb[neg])
        reg = l2_reg(self.user_emb(users), self.item_emb(pos), self.item_emb(neg), batch_size=users.size(0))
        return bpr + self.reg * reg


class SGL(Recommender):
    """Wu et al., SIGIR 2021 (edge-dropout variant)."""

    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3, drop_rate=0.1, temp=0.2,
                 reg=1e-3, lam=0.1):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.drop_rate, self.temp, self.reg, self.lam = drop_rate, temp, reg, lam
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def edge_dropout(self, adj):
        adj = adj.coalesce()
        indices, values = adj.indices(), adj.values()
        mask = torch.rand(values.size(0), device=values.device) > self.drop_rate
        return torch.sparse_coo_tensor(indices[:, mask], values[mask], adj.size(),
                                       check_invariants=False).coalesce()

    def forward(self, ui_adj):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_emb = lightgcn_propagate(ui_adj, ego, self.n_layers)
        return all_emb[: self.n_users], all_emb[self.n_users:]

    def ssl_loss(self, users, pos_items, ui_adj):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb1 = lightgcn_propagate(self.edge_dropout(ui_adj), ego, self.n_layers)
        emb2 = lightgcn_propagate(self.edge_dropout(ui_adj), ego, self.n_layers)
        loss_u = info_nce(emb1[: self.n_users][users], emb2[: self.n_users][users], self.temp)
        loss_i = info_nce(emb1[self.n_users:][pos_items], emb2[self.n_users:][pos_items], self.temp)
        return (loss_u + loss_i) / 2

    def embeddings(self, graphs: Graphs):
        return self.forward(graphs.ui)

    def loss(self, users, pos, neg, graphs: Graphs):
        user_emb, item_emb = self.forward(graphs.ui)
        bpr = bpr_loss(user_emb[users], item_emb[pos], item_emb[neg])
        reg = l2_reg(self.user_emb(users), self.item_emb(pos), self.item_emb(neg), batch_size=users.size(0))
        return bpr + self.reg * reg + self.lam * self.ssl_loss(users, pos, graphs.ui)


class GCMC(Recommender):
    """Graph Convolutional Matrix Completion (van den Berg et al., 2017), adapted to top-K ranking
    with a BPR loss and a single propagation step."""

    def __init__(self, n_users, n_items, emb_dim=64, hidden_dim=64, dropout=0.1, reg=1e-4):
        super().__init__()
        self.n_users, self.n_items, self.dropout, self.reg = n_users, n_items, dropout, reg
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.W_user = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W_item = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.dense_user = nn.Linear(hidden_dim, hidden_dim)
        self.dense_item = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.W_user.weight)
        nn.init.xavier_uniform_(self.W_item.weight)
        nn.init.xavier_uniform_(self.dense_user.weight)
        nn.init.xavier_uniform_(self.dense_item.weight)

    def forward(self, ui_adj):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        agg = torch.sparse.mm(ui_adj, ego)
        u_h = F.relu(self.W_user(agg[: self.n_users]))
        i_h = F.relu(self.W_item(agg[self.n_users:]))
        u_h = F.dropout(u_h, p=self.dropout, training=self.training)
        i_h = F.dropout(i_h, p=self.dropout, training=self.training)
        return self.dense_user(u_h), self.dense_item(i_h)

    def embeddings(self, graphs: Graphs):
        return self.forward(graphs.ui)

    def loss(self, users, pos, neg, graphs: Graphs):
        user_emb, item_emb = self.forward(graphs.ui)
        bpr = bpr_loss(user_emb[users], item_emb[pos], item_emb[neg])
        reg = l2_reg(self.user_emb(users), self.item_emb(pos), self.item_emb(neg), batch_size=users.size(0))
        return bpr + self.reg * reg


class NeuMF(Recommender):
    """Neural Matrix Factorization (He et al., WWW 2017): GMF + MLP."""

    def __init__(self, n_users, n_items, emb_dim=64, mlp_layers=(128, 64, 32), reg=1e-4):
        super().__init__()
        mlp_layers = list(mlp_layers)
        self.n_users, self.n_items, self.reg = n_users, n_items, reg
        self.user_gmf = nn.Embedding(n_users, emb_dim)
        self.item_gmf = nn.Embedding(n_items, emb_dim)
        self.user_mlp = nn.Embedding(n_users, emb_dim)
        self.item_mlp = nn.Embedding(n_items, emb_dim)

        layers, in_dim = [], emb_dim * 2
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.prediction = nn.Linear(emb_dim + mlp_layers[-1], 1, bias=False)

        nn.init.normal_(self.user_gmf.weight, std=0.01)
        nn.init.normal_(self.item_gmf.weight, std=0.01)
        nn.init.normal_(self.user_mlp.weight, std=0.01)
        nn.init.normal_(self.item_mlp.weight, std=0.01)

    def forward(self, users, items):
        gmf_out = self.user_gmf(users) * self.item_gmf(items)
        mlp_out = self.mlp(torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1))
        return self.prediction(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze(-1)

    def loss(self, users, pos, neg, graphs: Graphs | None = None):
        bpr = -F.logsigmoid(self.forward(users, pos) - self.forward(users, neg)).mean()
        reg = l2_reg(self.user_gmf(users), self.item_gmf(pos), self.item_gmf(neg),
                     self.user_mlp(users), self.item_mlp(pos), self.item_mlp(neg),
                     batch_size=users.size(0))
        return bpr + self.reg * reg

    @torch.no_grad()
    def scorer(self, graphs: Graphs | None = None):
        all_items = torch.arange(self.n_items, device=self.user_gmf.weight.device)

        def score(users: torch.Tensor) -> torch.Tensor:
            out = []
            for u in users.tolist():  # one forward pass over all items per user
                u_rep = torch.full((self.n_items,), u, dtype=torch.long, device=all_items.device)
                out.append(self.forward(u_rep, all_items))
            return torch.stack(out, dim=0)

        return score


class MFBPR(Recommender):
    """Matrix factorisation with BPR loss (Rendle et al., UAI 2009)."""

    def __init__(self, n_users, n_items, emb_dim=64, reg=1e-3):
        super().__init__()
        self.reg = reg
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self):
        return self.user_emb.weight, self.item_emb.weight

    def embeddings(self, graphs: Graphs | None = None):
        return self.forward()

    def loss(self, users, pos, neg, graphs: Graphs | None = None):
        u_e, p_e, n_e = self.user_emb(users), self.item_emb(pos), self.item_emb(neg)
        return bpr_loss(u_e, p_e, n_e) + self.reg * l2_reg(u_e, p_e, n_e, batch_size=users.size(0))
