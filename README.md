# LLM-MGCL: POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment

Official implementation of **LLM-MGCL**, a graph-based point-of-interest (POI) recommender that
extends LightGCN with two additional item-item views and aligns them with contrastive learning:

| Graph | Nodes | Edges | Edge weight |
|---|---|---|---|
| **G_UI** | users + items | training interactions | star rating, `(stars − 1) / 4`, clamped at 0.1 |
| **G_SEM** | items | top-10 most similar items within 50 km (cosine > 0) | cosine similarity of LLM-generated photo summaries (`all-MiniLM-L6-v2`) |
| **G_GEO** | items | top-10 nearest items within 50 km (Haversine) | `1 / (1 + distance_km)` |

The three graphs are propagated in parallel with LightGCN (3 / 2 / 2 layers) from shared ego
embeddings. The item representation is the sum of the three views. The model is trained with BPR,
L2 regularisation and a bidirectional InfoNCE loss that aligns the collaborative view with the
semantic and geographic views (`λ = 0.05`, `τ = 0.2`).


## Repository structure

```
├── configs/default.yaml          all hyper-parameters and paths (values used in the paper)
├── src/llm_mgcl/
│   ├── data.py                   loading, user filtering, per-user 80/10/10 split, id mappings
│   ├── graphs/                   G_UI, G_SEM, G_GEO construction + symmetric normalisation
│   ├── models/
│   │   ├── llm_mgcl.py           LLM-MGCL (incl. ablation switches)
│   │   ├── baselines.py          LightGCN, NGCF, SGL, GC-MC, NeuMF, MF-BPR
│   │   └── knn.py                ItemKNN, UserKNN
│   ├── sampler.py                BPR triple sampler
│   ├── trainer.py                shared training loop with early stopping
│   ├── evaluate.py               full-ranking Recall@K / NDCG@K (one protocol for all models)
│   ├── cold_start.py             item-popularity buckets
│   └── pipeline.py               loading processed data and trained checkpoints
├── scripts/
│   ├── build_graphs.py           step 1 - preprocessing + graph construction
│   ├── train.py                  step 2 - training (any model / ablation variant, all seeds)
│   ├── compare_models.py         step 3 - Table 1, ablation table, significance tests
│   ├── cold_start.py             step 4 - cold-start analysis
│   ├── make_figures.py           step 5 - figures
│   ├── run_all.sh                all of the above
│   └── import_colab_checkpoints.py
├── notebooks/01_data_and_graph_exploration.ipynb   statistics and visual sanity checks
└── tests/test_pipeline.py        end-to-end smoke test on synthetic data
```

## Installation

```bash
git clone https://github.com/wwebpy/LLM-MGCL.git && cd llm-mgcl
python -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

We use the [Yelp Multimodal Recommendation dataset](https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation)
(`review.csv`, `business.csv`, `photo_summay.csv`). It is read directly from the Hugging Face Hub;
if access requires authentication, run `huggingface-cli login` once (or set `HF_TOKEN`).
To work offline, download the three CSV files and pass the folder with `--raw <dir>`.

Preprocessing: users with fewer than 3 reviews are removed. Each user's interactions are shuffled
(`random_state=42`) and split 80 / 10 / 10 into train / validation / test. User and item ids, and
all graphs, are built from the training split only.

## Reproducing the paper

```bash
bash scripts/run_all.sh
```

or step by step:

```bash
python scripts/build_graphs.py                                   # data/processed/
python scripts/train.py --model llm_mgcl                          # seeds 42, 123, 2024
python scripts/train.py --model llm_mgcl --variant wo_cl          # ablations: wo_cl | wo_sem | wo_geo
python scripts/train.py --model lightgcn                          # lightgcn | sgl | ngcf | gcmc | neumf | mf_bpr
python scripts/compare_models.py                                  # ItemKNN / UserKNN are evaluated here
python scripts/cold_start.py
python scripts/make_figures.py
```

| Paper | Script | Output |
|---|---|---|
| Model comparison (Table 1) | `compare_models.py` | `results/table1_model_comparison.csv` |
| Significance tests | `compare_models.py` | `results/significance_tests.csv` |
| Ablation study | `train.py --variant …` + `compare_models.py` | `results/table_ablation.csv` |
| Cold-start analysis | `cold_start.py` | `results/cold_start_buckets.csv` |
| Cold-start figures | `make_figures.py` | `results/figures/*.pdf` |

**Training setup (all trainable models):** Adam (lr 1e-3), StepLR (step 20, γ 0.5), batch size
2048, up to 100 epochs. Validation Recall@20 is computed every 2 epochs, with early stopping after
8 evaluations without improvement. The best validation checkpoint is used for testing.
Embedding size is 64. Model-specific hyper-parameters are in `configs/default.yaml`.

**Evaluation protocol:** full ranking over all items. A user's training items are excluded, and
on the test split the user's validation items are excluded as well. Table 1 and the significance
tests use the users evaluated for all models. For each seed we average over users, then report
mean ± std over the 3 seeds. The significance tests use per-user metrics averaged over seeds,
with a paired t-test and a Wilcoxon signed-rank test, Holm–Bonferroni corrected.

**Cold start:** items are bucketed by their number of training interactions into cold (≤ 20),
warm (21–37) and hot (> 37), which are the 33rd and 66th percentiles. Recall@20 is computed on
the test items of each bucket.


## Pretrained checkpoints

_Link to the checkpoints (e.g. Hugging Face Hub / Zenodo)._ To evaluate checkpoints produced by
the original experiment notebook without retraining:

```bash
python scripts/build_graphs.py
python scripts/import_colab_checkpoints.py --src /path/to/GNN_Paper_Models
python scripts/compare_models.py 
python scripts/cold_start.py
```

## Reproducibility notes

* All randomness (model initialisation, BPR sampling, SGL edge dropout, NGCF / GC-MC dropout) is
  seeded via `set_seed(seed)` with seeds 42, 123 and 2024.
* The BPR sampler draws positives from a sorted list, so sampling does not depend on Python's hash
  seed. The original notebook iterated over Python sets, so a retrained model matches the
  published numbers statistically (within seed variance) but not bit for bit. Use the released
  checkpoints for exact numbers.
* Sparse matrix products on GPU are not bit-wise deterministic, so re-runs can differ slightly in
  the last decimals.
* Mutual k-NN pairs appear twice in the G_SEM / G_GEO edge lists. Their weights are summed during
  normalisation, as in the original implementation.
* Running `pytest` trains every model for two epochs on a small synthetic dataset (CPU, about 1.5 min).

## Citation

```bibtex
@misc{tamer2026poirecommendationllmaugmentedmultigraph,
      title={POI Recommendation with LLM-Augmented Multi-Graph Learning and Contrastive Alignment}, 
      author={Burak Tamer and Wolfram Höpken and Zehui Wang},
      year={2026},
      eprint={2608.16407},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2608.16407}, 
}
```

## License

Code: MIT (see `LICENSE`). The Yelp data is subject to its own license and terms of use.
