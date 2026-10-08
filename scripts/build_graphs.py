"""Step 1: preprocess the dataset and build G_UI, G_SEM and G_GEO.

Writes to ``paths.processed_dir``:
    train.csv / val.csv / test.csv   per-user 80/10/10 split
    item_texts.csv                   photo-summary texts used for G_SEM
    sem_embeddings.npz               Sentence-Transformer embeddings (re-used on later runs)
    edges.pt                         edge lists + weights of all three graphs, user/item order

Usage:
    python scripts/build_graphs.py [--config configs/default.yaml] [--raw <dir or hf:// path>]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from llm_mgcl.data import load_raw, prepare_interactions  # noqa: E402
from llm_mgcl.graphs import build_item_texts, build_semantic_edges, build_ui_edges, encode_texts  # noqa: E402
from llm_mgcl.graphs.adjacency import item_edges_to_tensor  # noqa: E402
from llm_mgcl.graphs.geo import build_geo_edges, geo_weight  # noqa: E402
from llm_mgcl.pipeline import processed_files  # noqa: E402
from llm_mgcl.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw", default=None, help="override paths.raw_dataset")
    ap.add_argument("--recompute-embeddings", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    g = cfg["graphs"]
    files = processed_files(cfg["paths"]["processed_dir"])
    files["train"].parent.mkdir(parents=True, exist_ok=True)

    # 1) Interactions -----------------------------------------------------------------------
    print("Loading raw data ...")
    reviews, business, photo_summary = load_raw(args.raw or cfg["paths"]["raw_dataset"])
    data = prepare_interactions(reviews, cfg["data"]["min_interactions"], cfg["data"]["split"],
                                cfg["data"]["split_seed"])
    print(f"Users {data.n_users:,} | Items {data.n_items:,} | "
          f"Train {len(data.train_df):,} | Val {len(data.val_df):,} | Test {len(data.test_df):,} | "
          f"density {len(data.train_df) / (data.n_users * data.n_items):.2e}")
    for name in ("train", "val", "test"):
        getattr(data, f"{name}_df").to_csv(files[name], index=False)

    # 2) G_UI ------------------------------------------------------------------------------
    ui_ei, ui_ew = build_ui_edges(data.train_df, data.user2idx, data.item2idx, g["ui"]["star_weight_min"])
    print(f"G_UI: {ui_ei.size(1):,} directed edges")

    # 3) G_SEM -----------------------------------------------------------------------------
    texts = build_item_texts(photo_summary, set(data.item2idx), g["semantic"]["text_fields"])
    texts[["business_id", "text_combined"]].to_csv(files["item_texts"], index=False)
    business_ids = texts["business_id"].tolist()
    print(f"Items with photo-summary text: {len(texts):,} of {data.n_items:,}")

    cached = files["embeddings"]
    if cached.exists() and not args.recompute_embeddings:
        z = np.load(cached, allow_pickle=True)
        if list(z["business_ids"]) != business_ids:
            raise RuntimeError(f"{cached} does not match the current items; use --recompute-embeddings")
        embeddings = z["embeddings"]
        print(f"Loaded cached embeddings {embeddings.shape}")
    else:
        embeddings = encode_texts(texts["text_combined"].tolist(), g["semantic"]["encoder"])
        np.savez_compressed(cached, embeddings=embeddings, business_ids=np.array(business_ids, dtype=object))
        print(f"Encoded embeddings {embeddings.shape}")

    sem_edges = build_semantic_edges(embeddings, business_ids, business, k=g["semantic"]["k"],
                                     max_dist_km=g["semantic"]["max_dist_km"],
                                     batch_size=g["semantic"]["batch_size"])
    sem_ei, sem_ew = item_edges_to_tensor(sem_edges, data.item2idx)

    # 4) G_GEO -----------------------------------------------------------------------------
    geo_edges = build_geo_edges(business, data.item2idx, k=g["geo"]["k"],
                                max_dist_km=g["geo"]["max_dist_km"], chunk_size=g["geo"]["chunk_size"])
    geo_ei, geo_ew = item_edges_to_tensor(geo_edges, data.item2idx, geo_weight)

    torch.save({
        "users": list(data.user2idx), "items": list(data.item2idx),
        "ui_edge_index": ui_ei, "ui_edge_weight": ui_ew,
        "sem_edge_index": sem_ei, "sem_edge_weight": sem_ew,
        "geo_edge_index": geo_ei, "geo_edge_weight": geo_ew,
    }, files["edges"])
    print(f"G_SEM: {sem_ei.size(1):,} | G_GEO: {geo_ei.size(1):,} bidirectional edges")
    print(f"Saved to {files['edges'].parent}")


if __name__ == "__main__":
    main()
