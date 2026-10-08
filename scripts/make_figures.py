"""Step 5: paper figures from the result files of ``scripts/cold_start.py``.

    recall_cold_warm_hot_combined.{pdf,png}   Recall@20 over fine popularity bins (LLM-MGCL, SGL, LightGCN)
    ablation_{cold,warm,hot}.{pdf,png}        Recall@20 of the ablation variants per bucket

Usage:
    python scripts/make_figures.py [--config configs/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from llm_mgcl.utils import load_config  # noqa: E402

COLORS = {"LLM-MGCL": "#2166ac", "SGL": "#b2182b", "LightGCN": "#4d4d4d"}
STYLES = {"LLM-MGCL": "-", "SGL": "--", "LightGCN": ":"}
ABLATION_ORDER = ["LightGCN", "w/o CL", "w/o G_SEM", "w/o G_GEO", "LLM-MGCL"]
ABLATION_TICKS = ["LightGCN", "w/o CL", r"w/o $G_{SEM}$", r"w/o $G_{GEO}$", "LLM-MGCL\n(full)"]


def _save(fig, path: Path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  {path.with_suffix('.pdf')}")


def fine_bin_plot(res: dict, out: Path):
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.linewidth": 0.8, "axes.edgecolor": "#333333"})
    segments = res["segments"]
    fig, axes = plt.subplots(1, len(segments), figsize=(15, 4.2))
    for ax, (seg, names) in zip(np.atleast_1d(axes), segments.items()):
        x = np.arange(len(names))
        for m in ["LLM-MGCL", "SGL", "LightGCN"]:
            if m not in res["results"]:
                continue
            vals = [res["results"][m][b]["recall"] for b in names]
            if not any(vals):
                continue
            y = np.array([np.mean(v) if v else np.nan for v in vals])
            s = np.array([np.std(v) if v else 0.0 for v in vals])
            ax.plot(x, y, STYLES[m], color=COLORS[m], lw=1.8, marker="o", ms=5, label=m)
            ax.fill_between(x, y - s, y + s, color=COLORS[m], alpha=0.15)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
        ax.set_xlabel("Training interactions per item", fontsize=9)
        ax.set_title(f"{seg.capitalize()} bucket", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        ax2 = ax.twiny()  # second x-axis: number of items per bin
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"n={res['items_per_bin'][n]:,}" for n in names], fontsize=6.5, color="#666666")
        ax2.set_xlabel("Items per bin", fontsize=8, color="#666666", labelpad=8)
        ax2.tick_params(axis="x", colors="#666666", length=3)
        ax2.spines[["right"]].set_visible(False)

    first = np.atleast_1d(axes)[0]
    first.set_ylabel(f"Recall@{res['k']}", fontsize=9)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.tight_layout()
    _save(fig, out / "recall_cold_warm_hot_combined")


def ablation_bars(res: dict, out: Path):
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
                         "axes.edgecolor": "#333333", "xtick.direction": "out", "ytick.direction": "out",
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8, "figure.dpi": 130})
    base, hl = "#b8c4d0", "#31506e"
    variants = [v for v in ABLATION_ORDER
                if v in res["results"] and any(b["recall"] for b in res["results"][v].values())]
    ticks = [ABLATION_TICKS[ABLATION_ORDER.index(v)] for v in variants]
    x = np.arange(len(variants))

    for bucket, (lo, hi) in res["buckets"].items():
        if not all(res["results"][v][bucket]["recall"] for v in variants):
            print(f"  skip ablation_{bucket}: no test items in this bucket for some model")
            continue
        mean = [np.mean(res["results"][v][bucket]["recall"]) for v in variants]
        std = [np.std(res["results"][v][bucket]["recall"]) for v in variants]
        best = int(np.argmax(mean))
        hi_txt = r"$>$" + f"{int(lo) - 1}" if hi == float("inf") else (
            r"$\leq$" + f"{int(hi)}" if lo == 0 else f"{int(lo)}–{int(hi)}")
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.bar(x, mean, yerr=std, capsize=3, width=0.66,
               color=[hl if i == best else base for i in range(len(variants))],
               edgecolor="#2b2b2b", linewidth=0.7,
               error_kw={"elinewidth": 0.9, "ecolor": "#2b2b2b", "capthick": 0.9})
        ymax = max(m + s for m, s in zip(mean, std)) or 1.0
        for i, (m, s) in enumerate(zip(mean, std)):
            ax.text(i, m + s + ymax * 0.03, f"{m:.4f}", ha="center", va="bottom", fontsize=7, color="#2b2b2b")
        ax.set_title(f"{bucket.capitalize()} ({hi_txt}" + (" interactions)" if lo == 0 else ")"), fontsize=10, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(f"Recall@{res['k']}", fontsize=9)
        ax.set_ylim(0, ymax * 1.22)
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        _save(fig, out / f"ablation_{bucket}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = Path(cfg["paths"]["results_dir"])
    out = Path(cfg["paths"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print("Writing figures:")
    with open(res_dir / "cold_start_fine_bins.json") as f:
        fine_bin_plot(json.load(f), out)
    with open(res_dir / "cold_start_buckets.json") as f:
        ablation_bars(json.load(f), out)


if __name__ == "__main__":
    main()
