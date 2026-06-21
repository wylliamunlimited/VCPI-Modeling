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
uv run python src/vcpi_ml/experiments/baseline.py   # per-gene mean → ~0.6119 wMSE
```

The pipeline is `data → expression → model(fit/predict) → evaluation`. Every
layer is verified against the contest package (expression == official to 0.0;
the full baseline == the contest's own per-gene-mean to 0.0).

## Repo layout

```
vcpi-ml/
├── pyproject.toml          # deps + package metadata (src layout)
├── src/vcpi_ml/
│   ├── sqlshell.py         # remote SQL REPL for exploration         ✅
│   ├── download.py         # data acquisition → data/raw/            ✅
│   ├── data.py             # read-side loaders + compound split      ✅
│   ├── expression.py       # counts → log2(CPM+1) target Y           ✅
│   ├── evaluation.py       # wide→long reshape + wMSE scorer wrapper  ✅
│   ├── features.py         # SMILES → Morgan fingerprint X            ✅
│   ├── models/
│   │   └── mean.py         # per-gene-mean baseline (fit/predict)     ✅
│   └── experiments/
│       └── baseline.py     # driver: runs the baseline end to end    ✅
├── data/raw/               # downloaded parquet, gitignored (populated ✅)
└── .env                    # TVC_TOKEN (gitignored)
```

## Roadmap

A difficulty ladder — each rung runnable, each teaches one concept.

**Track A — the bio model** (every rung scored on the same seed-0 split vs **0.6119**)
1. ✅ Load & look at the data (exploration, download)
2. ✅ Dumbest baseline: per-gene mean of train → **0.6119** wMSE (the floor)
3. 🔨 SMILES → Morgan fingerprint → Ridge regression (first chemistry-aware model)
4. ⬜ Plain MLP in PyTorch (first neural net)

**Track B — attention from scratch**
5. ⬜ Hand-write self-attention (Q/K/V + softmax) on a toy tensor
6. ⬜ Tiny char-level transformer over SMILES strings
7. ⬜ Understand ChemBERTa as a frozen feature extractor

Scoring uses the contest package's `score_compounds` / `load_gene_filter` /
`load_weights_matrix` (wrapped in `evaluation.py`); submissions are a parquet of
`(compound, gene_id, predicted_expression)` rows.
