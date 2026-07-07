# VCPI Chemical Compound to THP-1 Cell Gene Expression Modeling

My from-scratch implementation for the [VCPI 2026 prediction contest](https://github.com/virtualcell-vcpi/vcpi-prediction-contest-2026).
This is a learning repo: I'm building the whole pipeline myself — data
acquisition, exploration, features, and models — to learn hands-on ML for
biology.

## The task

Given a **drug compound's chemical structure** (a SMILES string + computed
molecular properties), predict **how each gene's expression changes** in THP-1
cells (a human monocyte line) 24 hours after exposure at 10 µM.

It's a **multi-output regression**: one compound in, a vector of ~13,000
per-gene expression values out.

- **Input (X):** a compound — SMILES + RDKit descriptors (`chemistry` table).
- **Output (Y):** per-gene expression, defined precisely as
  `mean over replicates of log2(CPM + 1)`, where
  `CPM = 1e6 * gene_count / total_counts_in_sample`.
- **Train:** ~14,000 compounds. **Test:** ~1,000 held-out compounds the model
  never saw — so the real challenge is **generalizing to novel chemistry**.
- **Metric:** weighted MSE (`wMSE`), lower is better. The per-(compound, gene)
  weights reward predicting each drug's *distinctive* expression changes, not
  the genes that barely move. (Baseline: predicting every gene's training-set
  average. The contest documents ~0.507 but never published its 200-compound
  split, so the reproducible bar here is **0.6119** on a fixed seed-0 split —
  verified identical to the contest's own per-gene-mean baseline to 0.0.)

### How the data connects

```
chemistry (1 row/compound)   metadata (1 row/sample)      counts (genes × samples)
  compound, SMILES,   ──join──  user_compound_id,   ──maps──  each sample is a column
  mol_weight, log_p     on      sequenced_id (a          to    of raw UMI counts
  ...                 compound  single measurement)            per gene
       │                                                              │
       └──────────► becomes X                    Y is built from ◄────┘
                                                 the counts (avg over replicates)
```

Each compound has several **replicate** samples; the target averages over them
to cancel measurement noise.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a `TVC_TOKEN` for the VCPI client.

```bash
uv sync                          # install deps into ./.venv
cp ../vcpi-prediction-contest-2026/.env .env   # token (gitignored)
```

The token is loaded at runtime via `uv run --env-file .env ...` — never
hardcoded, never committed.

## Exploring the data (no download needed)

The VCPI client can run SQL against the `metadata` and `chemistry` tables
**remotely** — nothing hits disk. `src/vcpi_ml/sqlshell.py` is a small REPL
around `vcpi.query()`:

```bash
uv run --env-file .env python src/vcpi_ml/sqlshell.py
```

```sql
sql> DESCRIBE metadata;                          -- see the columns (DuckDB syntax)
sql> SELECT user_compound_id, COUNT(*) AS n      -- replicates per compound
     FROM metadata GROUP BY 1 ORDER BY n DESC;
```

Tables available remotely: `metadata`, `chemistry` (join on `compound`).
The gene **counts** matrix is *not* queryable remotely — it requires the full
download below.

## Downloading the data

Pulls the 3 training releases + the scoring weights into `data/raw/`
(~1.4 GB counts, ~358 MB weights; peak RAM ~22 GB, a few minutes):

```bash
uv run --env-file .env python src/vcpi_ml/download.py
```

Produces `train_counts.parquet`, `train_metadata.parquet`,
`train_chemistry.parquet`, `weights.parquet` in `data/raw/`.

## Running a model

Each experiment composes the library layers into one runnable script:

```bash
uv run python src/vcpi_ml/experiments/baseline.py   # per-gene mean
uv run python src/vcpi_ml/experiments/ridge.py      # Morgan fingerprint → Ridge
```

The pipeline is `data → expression → model(fit/predict) → evaluation`. Every
layer is verified against the contest package (expression == official to 0.0;
the full baseline == the contest's own per-gene-mean to 0.0).

### Results (seed-0, 200-compound validation split)

| model | wMSE | vs baseline |
|---|---|---|
| per-gene mean (baseline / floor) | 0.6119 | — |
| Ridge, Morgan fingerprint, α=0.1 | 0.5796 | −0.032 |
| Ridge, α=10 | 0.5700 | −0.042 |
| **Ridge, α=100** | **0.5674** | **−0.045 (−7.3%)** |
| MLP (2×512 ReLU), full-batch, lr=1e-3 | 0.6089 | −0.003 |
| MLP (2×512 ReLU), mini-batch=500, lr=1e-3 | 0.6031 | −0.009 (−1.4%) |
| **MLP (2×512 ReLU), grid-tuned: lr=1e-3, batch=128, wd=0** | **0.5685** | **−0.043 (−7.1%)** |

Chemistry beats the floor: a linear map from substructure fingerprints to
expression predicts better than ignoring the molecule (Ridge). The optimum α
is above 100 (score still improving there).

The MLP now **essentially ties Ridge**. A grid sweep over lr × epoch × batch ×
weight-decay (with early stopping on the train-loss plateau) lands at **0.5685**
with `lr=1e-3, batch=128, wd=0` — within 0.001 of Ridge's 0.5674 and −7.1% under
the floor. Clear patterns from the sweep: **weight decay only hurt** (every
`wd>0` config scored worse), **smaller batches (128) edged out larger ones**, and
the **higher learning rate (5e-3) started to degrade**. So the gap to Ridge
closed not with more epochs but with the right lr/batch and no regularization.
Next: beating Ridge meaningfully likely needs richer features or architecture,
not just more tuning.

<details>
<summary><b>Full MLP grid sweep</b> (wMSE; best = <b>0.5685</b>)</summary>

Each cell is the validation wMSE for that `(lr, epoch, batch, wd)`. Lower is
better; the floor is 0.6119 and Ridge is 0.5674.

**lr = 1e-3**

| batch | 1k, wd=0 | 1k, wd=1e-4 | 1k, wd=1e-3 | 2.5k, wd=0 | 2.5k, wd=1e-4 | 2.5k, wd=1e-3 |
|---|---|---|---|---|---|---|
| 128 | **0.5685** | 0.5740 | 0.6142 | 0.5811 | 0.5802 | 0.6272 |
| 256 | — | 0.5792 | 0.6189 | 0.5756 | 0.5879 | — |
| 512 | 0.5761 | 0.5879 | 0.6219 | 0.5709 | 0.5818 | 0.6307 |

**lr = 3e-3**

| batch | 1k, wd=0 | 1k, wd=1e-4 | 1k, wd=1e-3 | 2.5k, wd=0 | 2.5k, wd=1e-4 | 2.5k, wd=1e-3 |
|---|---|---|---|---|---|---|
| 128 | 0.5827 | 0.6035 | 0.6340 | 0.5824 | — | 0.6307 |
| 256 | 0.5959 | — | 0.6463 | 0.5781 | 0.5823 | 0.6512 |
| 512 | 0.5729 | — | 0.6475 | 0.5769 | 0.5774 | 0.6318 |

**lr = 5e-3**

| batch | 1k, wd=0 | 1k, wd=1e-4 | 1k, wd=1e-3 | 2.5k, wd=0 | 2.5k, wd=1e-4 | 2.5k, wd=1e-3 |
|---|---|---|---|---|---|---|
| 128 | 0.6147 | 0.5947 | 0.6217 | 0.5883 | 0.5887 | 0.6487 |
| 256 | 0.5761 | 0.5768 | 0.6403 | 0.5825 | 0.5924 | 0.6577 |
| 512 | — | 0.6381 | n/r | 0.5770 | 0.5945 | 0.7211 |

`—` = score lost to a terminal display glitch (the config ran, the number
scrolled off); `n/r` = not run (interrupted). Every `wd=1e-3` column is the
worst in its block, and `wd=0` wins almost everywhere — regularization only
hurt at this scale.

</details>

## Repo layout

```
vcpi-ml/
├── pyproject.toml          # deps + package metadata (src layout)
├── src/vcpi_ml/
│   ├── sqlshell.py         # remote SQL REPL for exploration         ✅
│   ├── download.py         # data acquisition → data/raw/            ✅
│   ├── data.py             # loaders + compound split + SMILES tokenizer ✅
│   ├── expression.py       # counts → log2(CPM+1) target Y           ✅
│   ├── evaluation.py       # wide→long reshape + wMSE scorer wrapper  ✅
│   ├── features.py         # SMILES → Morgan fingerprint X            ✅
│   ├── device.py           # shared torch device (cuda→mps→cpu)       ✅
│   ├── models/
│   │   ├── mean.py         # per-gene-mean baseline (fit/predict)     ✅
│   │   ├── ridge.py        # Morgan fingerprint → Ridge (fit/predict) ✅
│   │   ├── mlp.py          # PyTorch MLP (fit/predict + train loop)   ✅
│   │   └── smiles_transformer.py  # transformer + fit/predict wrapper      🔨
│   └── experiments/
│       ├── baseline.py     # driver: per-gene-mean baseline           ✅
│       ├── ridge.py        # driver: Ridge (+ alpha sweep)            ✅
│       ├── mlp.py          # driver: MLP (+ hyperparameter grid)      ✅
│       └── smiles_transformer.py  # driver: transformer (coord. search) 🔨
├── notebooks/
│   ├── attention.py            # self-attention from scratch, toy tensor ✅
│   └── attention_explained.md  # study notes: Q/K/V, dims, multi-head    ✅
├── data/raw/               # downloaded parquet, gitignored (populated ✅)
└── .env                    # TVC_TOKEN (gitignored)
```

## Roadmap

A difficulty ladder — each rung runnable, each teaches one concept.

**Track A — the bio model** (every rung scored on the same seed-0 split vs **0.6119**)
1. ✅ Load & look at the data (exploration, download)
2. ✅ Dumbest baseline: per-gene mean of train → **0.6119** wMSE (the floor)
3. ✅ SMILES → Morgan fingerprint → Ridge → **0.5674** (chemistry beats the floor)
4. ✅ Plain MLP in PyTorch (first neural net) → **0.5685** (grid-tuned:
   lr=1e-3, batch=128, no weight decay) — essentially ties Ridge; beating it
   likely needs richer features or architecture, not more tuning

**Track B — attention from scratch**
5. ✅ Hand-write self-attention (Q/K/V + softmax) on a toy tensor
   (`notebooks/attention.py` + `attention_explained.md`)
6. 🔨 Char-level SMILES transformer, hand-written from scratch — char
   tokenizer, multi-head attention (packed Q/K/V + head reshape), Add & Norm
   encoder blocks, padding mask + masked mean-pool, regression head, and the
   TransformerModel fit/predict wrapper + coordinate-search driver are all
   built and pass end-to-end fit/predict smoke tests. **Next:** run the sweep
   and score vs the 0.5674 Ridge bar.
7. ⬜ Understand ChemBERTa as a frozen feature extractor

Scoring uses the contest package's `score_compounds` / `load_gene_filter` /
`load_weights_matrix` (wrapped in `evaluation.py`); submissions are a parquet of
`(compound, gene_id, predicted_expression)` rows.
