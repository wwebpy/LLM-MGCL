"""Convert checkpoints written by the original Colab notebook into this repository's layout.

The model classes keep the original parameter names, so the old ``model_state`` dicts load as-is.
The script searches ``--src`` recursively for the original file names, e.g.

    llm_mgcl_geo_seed42.pt, ablation_wocl_seed42.pt, ablation_wosem_seed42.pt, ablation_wogeo_seed42.pt,
    lightgcn_baseline_seed42.pt, ngcf_seed42.pt, sgl_seed42.pt, gcmc_seed42.pt, neumf_seed42.pt, mfbpr_seed42.pt

and writes them to ``checkpoints/<run>/<run>_seed<seed>.pt``. Afterwards ``compare_models.py``
and ``cold_start.py`` can be run on the original models without retraining (the processed data
must be built with the same configuration).

Usage:
    python scripts/import_colab_checkpoints.py --src /path/to/GNN_Paper_Models
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from llm_mgcl.utils import checkpoint_path, load_config, save_checkpoint  # noqa: E402

# original file prefix -> (model, variant)
PREFIXES = {
    "llm_mgcl_geo": ("llm_mgcl", "full"),
    "ablation_wocl": ("llm_mgcl", "wo_cl"),
    "ablation_wosem": ("llm_mgcl", "wo_sem"),
    "ablation_wogeo": ("llm_mgcl", "wo_geo"),
    "lightgcn_baseline": ("lightgcn", "full"),
    "ngcf": ("ngcf", "full"),
    "sgl": ("sgl", "full"),
    "gcmc": ("gcmc", "full"),
    "neumf": ("neumf", "full"),
    "mfbpr": ("mf_bpr", "full"),
}
PATTERN = re.compile(r"^(?P<prefix>.+)_seed(?P<seed>\d+)\.pt$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with the original .pt files (searched recursively)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    n = 0
    for path in sorted(Path(args.src).rglob("*.pt")):
        m = PATTERN.match(path.name)
        if not m or m["prefix"] not in PREFIXES:
            continue
        model, variant = PREFIXES[m["prefix"]]
        seed = int(m["seed"])
        target = checkpoint_path(cfg, model, seed, variant)
        if target.exists() and not args.overwrite:
            print(f"skip (exists): {target}")
            continue
        old = torch.load(path, map_location="cpu", weights_only=False)
        state = old["model_state"] if "model_state" in old else old
        if state is None:
            print(f"skip (no weights): {path}")
            continue
        save_checkpoint(target, {
            "model": model, "variant": variant, "seed": seed,
            "model_config": old.get("model_config"),
            "model_state": state,
            "imported_from": str(path),
        })
        print(f"{path.name:<32} -> {target}")
        n += 1
    print(f"Imported {n} checkpoint(s).")


if __name__ == "__main__":
    main()
