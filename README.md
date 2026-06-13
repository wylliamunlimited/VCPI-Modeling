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
  average scores ~0.507 — that's the bar to beat.)

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
The gene **counts** matrix is *not* queryable remotely — it requires a full
download (not yet implemented; see roadmap).

## Repo layout

```
vcpi-ml/
├── pyproject.toml          # deps + package metadata (src layout)
├── src/vcpi_ml/
│   └── sqlshell.py         # remote SQL REPL for exploration  ✅
├── notebooks/              # exploration & modeling notebooks (empty so far)
├── data/raw/               # downloaded parquet, gitignored (empty so far)
└── .env                    # TVC_TOKEN (gitignored)
```

## Roadmap

A difficulty ladder — each rung runnable, each teaches one concept.
**Done:** remote SQL exploration (`sqlshell.py`).

**Next — data acquisition**
- `download.py`: pull the 3 training releases + scoring weights into `data/raw/`
  (~1.4 GB counts, ~358 MB weights; peak RAM ~22 GB).

**Track A — the bio model**
1. Load & look at the data (exploration)
2. Dumbest baseline: per-gene mean of train → score it (~0.507)
3. SMILES → Morgan fingerprint → Ridge regression (first real model)
4. Plain MLP in PyTorch (first neural net)

**Track B — attention from scratch**
5. Hand-write self-attention (Q/K/V + softmax) on a toy tensor
6. Tiny char-level transformer over SMILES strings
7. Understand ChemBERTa as a frozen feature extractor

Scoring will use the contest package's `score_compounds` / `load_gene_filter` /
`load_weights_matrix`; submissions are a parquet of
`(compound, gene_id, predicted_expression)` rows.
