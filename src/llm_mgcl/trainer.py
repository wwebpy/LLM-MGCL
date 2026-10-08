"""Shared training loop: BPR sampling, Adam + StepLR, early stopping on validation Recall@K."""

from __future__ import annotations

import time

import torch

from .data import Interactions
from .evaluate import evaluate
from .graphs import Graphs
from .models import Recommender
from .sampler import BPRSampler


def _score_fn(model: Recommender, graphs: Graphs):
    model.eval()
    return model.scorer(graphs)


def train(model: Recommender, data: Interactions, graphs: Graphs, cfg: dict, device: torch.device,
          log_prefix: str = "") -> tuple[Recommender, dict]:
    """Train ``model`` in place and return it with the best validation weights loaded.

    Returns ``(model, info)`` where ``info`` contains the best validation Recall@K, the epoch it
    was reached at and the number of epochs run.
    """
    t = cfg["training"]
    ks = tuple(cfg["evaluation"]["ks"])
    k_sel = t["early_stop_k"]
    sampler = BPRSampler(data, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=t["lr_step_size"], gamma=t["lr_gamma"])
    steps_per_epoch = max(1, len(data.train_df) // t["batch_size"])

    best_val, best_state, best_epoch, no_improve, epoch = -1.0, None, 0, 0, 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()
        for _ in range(steps_per_epoch):
            users, pos, neg = sampler.sample(t["batch_size"])
            optimizer.zero_grad()
            loss = model.loss(users, pos, neg, graphs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        msg = f"{log_prefix}Epoch {epoch:03d} | loss {total_loss / steps_per_epoch:.4f} | {time.time() - t0:.1f}s"

        if epoch % t["eval_every"] == 0:
            val = evaluate(_score_fn(model, graphs), data, "val", (k_sel,), t["eval_batch_size"], device)
            val_recall = val[f"recall@{k_sel}"]
            msg += f" | val R@{k_sel} {val_recall:.4f} N@{k_sel} {val[f'ndcg@{k_sel}']:.4f}"
            if val_recall > best_val:
                best_val, best_epoch, no_improve = val_recall, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                msg += " | best"
            else:
                no_improve += 1
                msg += f" | no improvement ({no_improve}/{t['patience']})"
            print(msg)
            if no_improve >= t["patience"]:
                print(f"{log_prefix}Early stopping at epoch {epoch}")
                break
        else:
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_recall": best_val, "best_epoch": best_epoch, "epochs_run": epoch,
                   "metric_ks": list(ks)}


def test_metrics(model: Recommender, data: Interactions, graphs: Graphs, cfg: dict, device) -> dict:
    ks = tuple(cfg["evaluation"]["ks"])
    return evaluate(_score_fn(model, graphs), data, "test", ks, cfg["training"]["eval_batch_size"], device)
