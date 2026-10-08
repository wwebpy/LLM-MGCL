"""Step 4: cold-start analysis - Recall@20 / NDCG@20 stratified by item popularity.

Items are bucketed by their number of training interactions (configurable in
``cold_start.buckets``; default cold <= 20 < warm <= 37 < hot). Evaluated models: LightGCN, SGL,
LLM-MGCL and its ablation variants.

Outputs (``results/``):
    cold_start_buckets.csv / .json   cold / warm / hot (mean ± std over seeds, per-seed values)
    cold_start_fine_bins.json        fine bins used for the line plot (``cold_start.fine_bins``)

Usage:
    python scripts/cold_start.py [--config configs/default.yaml]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from llm_mgcl.cold_start import assign_buckets, coarse_bins, fine_bins, items_per_bin  # noqa: E402
from llm_mgcl.evaluate import bucket_metrics  # noqa: E402
from llm_mgcl.pipeline import load_dataset, load_trained_model  # noqa: E402
from llm_mgcl.utils import get_device, load_config, save_json  # noqa: E402

# label -> (model, variant)
RUNS = {
    "LightGCN": ("lightgcn", None),
    "SGL": ("sgl", None),
    "w/o CL": ("llm_mgcl", "wo_cl"),
    "w/o G_SEM": ("llm_mgcl", "wo_sem"),
    "w/o G_GEO": ("llm_mgcl", "wo_geo"),
    "LLM-MGCL": ("llm_mgcl", None),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    ds = load_dataset(cfg, device)
    seeds, k = cfg["training"]["seeds"], cfg["cold_start"]["k"]
    bs = cfg["training"]["eval_batch_size"]
    out_dir = Path(cfg["paths"]["results_dir"])

    coarse = coarse_bins(cfg)
    fine = fine_bins(cfg)
    fine_all = [b for seg in fine.values() for b in seg]
    coarse_map = assign_buckets(ds.data, coarse)
    fine_map = assign_buckets(ds.data, fine_all)
    coarse_names = [b[0] for b in coarse]
    fine_names = [b[0] for b in fine_all]

    counts = items_per_bin(ds.data, coarse)
    print("Items per bucket: " + ", ".join(f"{b}={n:,}" for b, n in counts.items()))

    coarse_res = {label: {b: {"recall": [], "ndcg": []} for b in coarse_names} for label in RUNS}
    fine_res = {label: {b: {"recall": [], "n_users": []} for b in fine_names} for label in RUNS}

    for label, (name, variant) in RUNS.items():
        for s in seeds:
            model = load_trained_model(name, s, cfg, ds.data, device, variant)
            if model is None:
                continue
            score = model.scorer(ds.graphs)
            res = bucket_metrics(score, ds.data, coarse_map, coarse_names, "test", k, bs, device)
            for b in coarse_names:
                if res[b]:
                    coarse_res[label][b]["recall"].append(res[b]["recall"])
                    coarse_res[label][b]["ndcg"].append(res[b]["ndcg"])
            res = bucket_metrics(score, ds.data, fine_map, fine_names, "test", k, bs, device)
            for b in fine_names:
                if res[b]:
                    fine_res[label][b]["recall"].append(res[b]["recall"])
                    fine_res[label][b]["n_users"].append(res[b]["n_users_with_gt"])
            print(f"{label} | seed {s} | " + " ".join(
                f"{b}={coarse_res[label][b]['recall'][-1]:.4f}" for b in coarse_names
                if coarse_res[label][b]["recall"]))

    # Coarse buckets -----------------------------------------------------------------------
    rows = []
    for label, buckets in coarse_res.items():
        if not any(buckets[b]["recall"] for b in coarse_names):
            continue
        row = {"model": label}
        for b in coarse_names:
            for metric in ("recall", "ndcg"):
                vals = buckets[b][metric]
                row[f"{metric}@{k}_{b}"] = np.mean(vals) if vals else np.nan
                row[f"{metric}@{k}_{b}_std"] = np.std(vals) if vals else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cold_start_buckets.csv", index=False)
    save_json({"k": k, "buckets": {n: [lo, hi] for n, lo, hi in coarse}, "items_per_bucket": counts,
               "results": coarse_res}, out_dir / "cold_start_buckets.json")

    print(f"\n=== Recall@{k} per popularity bucket (mean ± std over seeds) ===")
    for _, r in df.iterrows():
        print(f"{r['model']:<12} " + "  ".join(
            f"{b}: {r[f'recall@{k}_{b}']:.4f} ± {r[f'recall@{k}_{b}_std']:.4f}" for b in coarse_names))

    if "LightGCN" in set(df["model"]):
        base = df.set_index("model").loc["LightGCN"]
        print("\nRelative change vs. LightGCN:")
        for _, r in df[df["model"] != "LightGCN"].iterrows():
            parts = []
            for b in coarse_names:
                lg, v = base[f"recall@{k}_{b}"], r[f"recall@{k}_{b}"]
                parts.append(f"{b}: {(v - lg) / lg * 100:+.1f}%" if lg > 0 else f"{b}: {v - lg:+.4f} (abs.)")
            print(f"  {r['model']:<12} " + "  ".join(parts))

    # Fine bins ----------------------------------------------------------------------------
    save_json({"k": k,
               "segments": {seg: [b[0] for b in bins] for seg, bins in fine.items()},
               "items_per_bin": items_per_bin(ds.data, fine_all),
               "results": fine_res}, out_dir / "cold_start_fine_bins.json")
    print(f"\nSaved to {out_dir}/cold_start_buckets.csv, cold_start_buckets.json, cold_start_fine_bins.json")


if __name__ == "__main__":
    main()
