"""Step 3: model comparison (Table 1), ablation table and significance tests.

Loads all trained checkpoints, computes per-user test metrics (train + val items masked), and
restricts every comparison to the users evaluated for *all* models.

* Table 1: per seed, the mean over the common users; reported as mean ± std over seeds.
* Significance: per-user metrics averaged over seeds; LLM-MGCL vs. each baseline with a paired
  t-test and a Wilcoxon signed-rank test, Holm-Bonferroni corrected over all
  (baseline x metric) comparisons, separately for each test.

Outputs (``results/``): table1_model_comparison.csv, table_ablation.csv, significance_tests.csv

Usage:
    python scripts/compare_models.py [--config configs/default.yaml]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from llm_mgcl.evaluate import metric_names, per_user_metrics  # noqa: E402
from llm_mgcl.models import DISPLAY_NAMES, KNN, TRAINABLE  # noqa: E402
from llm_mgcl.pipeline import load_dataset, load_trained_model  # noqa: E402
from llm_mgcl.utils import get_device, load_config  # noqa: E402

ABLATION_LABELS = {"wo_cl": "w/o CL", "wo_sem": "w/o G_SEM", "wo_geo": "w/o G_GEO"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    ds = load_dataset(cfg, device)
    ks = tuple(cfg["evaluation"]["ks"])
    bs = cfg["training"]["eval_batch_size"]
    seeds = cfg["training"]["seeds"]
    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def per_user(model):
        return per_user_metrics(model.scorer(ds.graphs), ds.data, "test", ks, bs, device)

    # Per-user metrics: label -> list (one dict per seed) --------------------------------
    peruser: dict[str, list[dict]] = {}
    for name in TRAINABLE:
        label = DISPLAY_NAMES[name]
        for s in seeds:
            model = load_trained_model(name, s, cfg, ds.data, device)
            if model is not None:
                print(f"{label} | seed {s}")
                peruser.setdefault(label, []).append(per_user(model))
    for name, cls in KNN.items():  # deterministic: a single run
        label = DISPLAY_NAMES[name]
        print(label)
        knn = cls(ds.data, **cfg["models"][name])
        peruser[label] = [per_user_metrics(knn.scorer(device), ds.data, "test", ks, bs, device)]

    ablation: dict[str, list[dict]] = {}
    for variant, label in ABLATION_LABELS.items():
        for s in seeds:
            model = load_trained_model("llm_mgcl", s, cfg, ds.data, device, variant)
            if model is not None:
                print(f"{label} | seed {s}")
                ablation.setdefault(label, []).append(per_user(model))

    if "LLM-MGCL" not in peruser:
        sys.exit("No LLM-MGCL checkpoints found - train the models first.")

    all_dicts = [d for dicts in list(peruser.values()) + list(ablation.values()) for d in dicts]
    common = sorted(set.intersection(*[set(d) for d in all_dicts]))
    print(f"\nCommon test users: {len(common):,}")
    names = metric_names(ks)

    def table(results: dict[str, list[dict]], label_col: str) -> pd.DataFrame:
        rows = []
        for label, dicts in results.items():
            per_seed = np.array([np.array([d[u] for u in common]).mean(axis=0) for d in dicts])
            mean, std = per_seed.mean(axis=0), per_seed.std(axis=0)
            row = {label_col: label, "n_seeds": len(dicts)}
            for j, m in enumerate(names):
                row[m] = mean[j]
                row[f"{m}_std"] = std[j] if len(dicts) > 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    t1 = table(peruser, "model").sort_values(f"recall@{max(ks)}", ascending=False)
    t1.to_csv(out_dir / "table1_model_comparison.csv", index=False)
    print("\n=== Table 1: model comparison (test, mean ± std over seeds) ===")
    print(_pretty(t1, "model", names))

    if ablation:
        ab = table({"LLM-MGCL (full)": peruser["LLM-MGCL"], **ablation}, "variant")
        ab.to_csv(out_dir / "table_ablation.csv", index=False)
        print("\n=== Ablation study ===")
        print(_pretty(ab, "variant", names))

    # Significance tests --------------------------------------------------------------------
    M = {label: np.array([[d[u] for u in common] for d in dicts]).mean(axis=0)
         for label, dicts in peruser.items()}
    rows, p_t, p_w = [], [], []
    for base in [label for label in peruser if label != "LLM-MGCL"]:
        for j, m in enumerate(names):
            a, b = M["LLM-MGCL"][:, j], M[base][:, j]
            t_stat, pt = stats.ttest_rel(a, b)
            try:
                _, pw = stats.wilcoxon(a, b)
            except ValueError:  # all differences zero
                pw = 1.0
            rows.append({"baseline": base, "metric": m, "llm_mgcl": a.mean(), "baseline_value": b.mean(),
                         "mean_diff": (a - b).mean(), "t": t_stat, "p_ttest": pt, "p_wilcoxon": pw})
            p_t.append(pt)
            p_w.append(pw)

    rej_t, p_t_adj, _, _ = multipletests(p_t, alpha=args.alpha, method="holm")
    rej_w, p_w_adj, _, _ = multipletests(p_w, alpha=args.alpha, method="holm")
    for r, pta, pwa, rt, rw in zip(rows, p_t_adj, p_w_adj, rej_t, rej_w):
        r["p_ttest_holm"], r["p_wilcoxon_holm"] = pta, pwa
        r["significant"] = "both" if (rt and rw) else ("t-test only" if rt else ("wilcoxon only" if rw else "n.s."))
    sig = pd.DataFrame(rows)
    sig.to_csv(out_dir / "significance_tests.csv", index=False)

    print("\n=== Significance: LLM-MGCL vs. baselines (paired t-test & Wilcoxon, Holm-corrected) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4g}", "display.width", 200):
        print(sig[["baseline", "metric", "llm_mgcl", "baseline_value", "mean_diff", "t",
                   "p_ttest_holm", "p_wilcoxon_holm", "significant"]].to_string(index=False))
    print(f"\nSaved tables to {out_dir}/")


def _pretty(df: pd.DataFrame, label_col: str, names: list[str]) -> str:
    out = df[[label_col]].copy()
    for m in names:
        out[m] = [f"{v:.4f}" + ("" if np.isnan(s) else f" ± {s:.4f}") for v, s in zip(df[m], df[f"{m}_std"])]
    return out.to_string(index=False)


if __name__ == "__main__":
    main()
