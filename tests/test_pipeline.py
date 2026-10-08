"""End-to-end smoke test on a tiny synthetic dataset (CPU, ~1 minute).

Runs the real scripts: build_graphs -> train (all models, 1 seed, 2 epochs) -> compare_models ->
cold_start -> make_figures. Pre-computed random embeddings replace the Sentence-Transformer.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_mgcl.data import prepare_interactions  # noqa: E402
from llm_mgcl.graphs import build_item_texts  # noqa: E402


def make_raw(raw_dir: Path, n_users=120, n_items=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for i in rng.choice(n_items, size=rng.integers(3, 15), replace=False):
            rows.append((f"u{u}", f"b{i}", float(rng.integers(1, 6))))
    pd.DataFrame(rows, columns=["user_id", "business_id", "stars"]).to_csv(raw_dir / "review.csv", index=False)
    # two "cities" so that the 50 km filter matters
    city = rng.integers(0, 2, n_items)
    lat = np.where(city == 0, 39.95, 36.17) + rng.normal(0, 0.05, n_items)
    lon = np.where(city == 0, -75.16, -86.78) + rng.normal(0, 0.05, n_items)
    pd.DataFrame({"business_id": [f"b{i}" for i in range(n_items)], "latitude": lat, "longitude": lon}) \
        .to_csv(raw_dir / "business.csv", index=False)
    pd.DataFrame({"business_id": [f"b{i}" for i in range(0, n_items, 2)] + ["b1"],
                  "summary": [f"cozy place {i % 5}" for i in range(0, n_items, 2)] + [""],
                  "keywords": [f"kw{i % 3}" for i in range(0, n_items, 2)] + [None]}) \
        .to_csv(raw_dir / "photo_summay.csv", index=False)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    raw = work / "raw"
    raw.mkdir()
    make_raw(raw)

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))
    cfg["paths"] = {k: str(work / Path(v)) for k, v in cfg["paths"].items()}
    cfg["paths"]["raw_dataset"] = str(raw)
    cfg["training"].update(seeds=[42], epochs=2, eval_every=1, batch_size=64)
    cfg_path = work / "config.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    # fake embeddings in the order build_graphs.py will use
    reviews = pd.read_csv(raw / "review.csv")
    data = prepare_interactions(reviews, cfg["data"]["min_interactions"], cfg["data"]["split"], 42)
    texts = build_item_texts(pd.read_csv(raw / "photo_summay.csv"), set(data.item2idx))
    proc = Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True)
    emb = np.random.default_rng(1).normal(size=(len(texts), 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez_compressed(proc / "sem_embeddings.npz", embeddings=emb,
                        business_ids=np.array(texts["business_id"].tolist(), dtype=object))
    return work, cfg_path, cfg


def run(script, *args):
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, f"{script} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
    return res.stdout


def test_end_to_end(workdir):
    work, cfg_path, cfg = workdir
    out = run("build_graphs.py", "--config", str(cfg_path))
    assert "G_SEM" in out

    for variant in ["full", "wo_cl", "wo_sem", "wo_geo"]:
        run("train.py", "--model", "llm_mgcl", "--variant", variant, "--config", str(cfg_path), "--device", "cpu")
    for model in ["lightgcn", "sgl", "ngcf", "gcmc", "neumf", "mf_bpr"]:
        run("train.py", "--model", model, "--config", str(cfg_path), "--device", "cpu")

    out = run("compare_models.py", "--config", str(cfg_path), "--device", "cpu")
    assert "Table 1" in out and "ItemKNN" in out and "UserKNN" in out
    res = Path(cfg["paths"]["results_dir"])
    t1 = pd.read_csv(res / "table1_model_comparison.csv")
    assert len(t1) == 9
    assert ((t1["recall@20"] >= 0) & (t1["recall@20"] <= 1)).all()
    assert len(pd.read_csv(res / "significance_tests.csv")) == 8 * 4
    assert len(pd.read_csv(res / "table_ablation.csv")) == 4

    run("cold_start.py", "--config", str(cfg_path), "--device", "cpu")
    run("make_figures.py", "--config", str(cfg_path))
    figs = Path(cfg["paths"]["figures_dir"])
    assert (figs / "recall_cold_warm_hot_combined.pdf").exists()
    assert (figs / "ablation_cold.pdf").exists()
