"""Step 2: train a model (or an LLM-MGCL ablation variant) for all seeds.

Saves one checkpoint per seed to ``checkpoints/<run>/<run>_seed<seed>.pt`` and the test metrics
(all test users) to ``results/runs/<run>.json``.

Examples:
    python scripts/train.py --model llm_mgcl
    python scripts/train.py --model llm_mgcl --variant wo_cl
    python scripts/train.py --model lightgcn --seeds 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_mgcl.models import TRAINABLE, build_model  # noqa: E402
from llm_mgcl.pipeline import load_dataset  # noqa: E402
from llm_mgcl.trainer import test_metrics, train  # noqa: E402
from llm_mgcl.utils import (checkpoint_path, get_device, load_config, mean_std, run_name,  # noqa: E402
                            save_checkpoint, save_json, set_seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(TRAINABLE))
    ap.add_argument("--variant", default="full", help="llm_mgcl only: full | wo_cl | wo_sem | wo_geo")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model != "llm_mgcl" and args.variant != "full":
        ap.error("--variant is only available for llm_mgcl")
    if args.model == "llm_mgcl" and args.variant not in cfg["ablations"]:
        ap.error(f"unknown variant, choose from {list(cfg['ablations'])}")

    device = get_device(args.device)
    seeds = args.seeds or cfg["training"]["seeds"]
    name = run_name(args.model, args.variant)
    ds = load_dataset(cfg, device)

    runs = []
    for seed in seeds:
        print(f"\n{'=' * 60}\n{name} | seed {seed} | device {device}\n{'=' * 60}")
        set_seed(seed)
        model = build_model(args.model, ds.data.n_users, ds.data.n_items, cfg, args.variant).to(device)
        model, info = train(model, ds.data, ds.graphs, cfg, device)
        metrics = test_metrics(model, ds.data, ds.graphs, cfg, device)
        print(f"{name} seed {seed} | " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        save_checkpoint(checkpoint_path(cfg, args.model, seed, args.variant), {
            "model": args.model,
            "variant": args.variant,
            "seed": seed,
            "model_config": {"n_users": ds.data.n_users, "n_items": ds.data.n_items,
                             **cfg["models"][args.model],
                             **(cfg["ablations"][args.variant] if args.model == "llm_mgcl" else {})},
            "model_state": model.state_dict(),
            "test_metrics": metrics,
            **info,
        })
        runs.append({"seed": seed, **metrics, **info})

    summary = {m: dict(zip(("mean", "std"), mean_std([r[m] for r in runs]))) for m in metrics}
    save_json({"run": name, "seeds": seeds, "runs": runs, "summary": summary},
              Path(cfg["paths"]["results_dir"]) / "runs" / f"{name}.json")

    print(f"\n{name} - test metrics over {len(seeds)} seed(s) (mean ± std)")
    for m, s in summary.items():
        print(f"  {m:<10} {s['mean']:.4f} ± {s['std']:.4f}")


if __name__ == "__main__":
    main()
