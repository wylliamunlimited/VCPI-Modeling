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
| SMILES transformer (from scratch), tuned: d_model=256, 4 layers, 4 heads, lr=3e-4 | 0.5890 | −0.023 (−3.7%) |
| Ridge on frozen ChemBERTa embeddings, α=10 | 0.5812 | −0.031 |
| **Ridge on frozen ChemBERTa embeddings, α=100** | **0.5794** | **−0.033 (−5.3%)** |
| **MLP on frozen ChemBERTa embeddings, grid-tuned: lr=1e-3, batch=256, wd=0** | **0.5727** | **−0.039 (−6.4%)** |

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

The from-scratch **SMILES transformer lands at 0.5890** — it clears the floor
(−3.7%) but **loses to Ridge/MLP** (0.5674/0.5685). This is the expected result,
not a bug, and it's the most interesting finding so far:

- **The signal is largely linear.** The MLP already told us this (it only tied
  Ridge). A transformer's whole edge is learning nonlinear, context-dependent
  features — wasted when structure→expression is mostly linear.
- **Transformers are data-hungry; ~12,800 molecules is tiny.** From scratch, it
  must *learn* chemistry from raw characters, while Morgan fingerprints hand-encode
  substructure priors for free. There isn't enough data to learn a better
  representation than the fingerprint already gives.
- **Capacity was still helping monotonically** (d_model 128→256 and depth 1→4
  both improved it), i.e. it's representation-**under**fitting, not overfitting —
  which points to *more data / a pretrained encoder*, exactly the rung-7 argument.
- **The coordinate search rides noise.** Re-running the *same* config gave 0.5983
  then 0.6039 (a ~0.006 swing from init/dropout/shuffle), so the per-knob "winners"
  are partly luck; 0.5890 is a real number but sits inside a ±0.005-ish band.

Takeaway: a hand-built transformer on 13k SMILES *can't* out-feature a Morgan
fingerprint. The natural next test is a **pretrained** SMILES encoder (ChemBERTa,
rung 7) — transfer the chemistry knowledge the local data lacks.

**And the surprise: it still doesn't beat Morgan.** Frozen ChemBERTa embeddings
(768-dim, mean-pooled, ZINC-pretrained) into the *same* Ridge land at **0.5794**
(α=100, still improving) — better than the from-scratch transformer (0.5890) and
well under the floor, but **above Morgan-Ridge's 0.5674**. So a transformer
pretrained on ~millions of molecules loses to a hand-designed fingerprint here.
Two honest reasons:

- **The task is largely linear, and Morgan is *already* a great linear basis.**
  Fingerprint bits are explicit substructure indicators — near-ideal features for
  a linear model. ChemBERTa's dense embedding is optimized for masked-token
  prediction, not for being linearly separable into expression, so Ridge extracts
  less from it per dimension.
- **Domain mismatch + frozen + mean-pool.** ChemBERTa-zinc was pretrained on ZINC
  drug-like molecules (may not match this compound library), it's *frozen* (never
  adapted to the task), and a plain mean-pool discards structure. The winning
  pipelines don't use it alone — they **fuse** it with fingerprints and PCA the
  target (see `AiBio/`); ChemBERTa *adds* signal on top of Morgan, it doesn't
  replace it.

Swapping Ridge for a **nonlinear MLP head on the same ChemBERTa embeddings**
helps — a grid-tuned MLP reaches **0.5727** (`lr=1e-3, batch=256, wd=0`), beating
ChemBERTa-Ridge's 0.5794. This is the *opposite* of the Morgan case (where the MLP
only *tied* Ridge): the dense ChemBERTa embedding is **not** as linearly separable
as fingerprint bits, so the nonlinearity actually earns its keep here. But 0.5727
still lands just above Morgan-Ridge's 0.5674. The weight-decay pattern repeats
hard — every `wd>0` config scored worse, several catastrophically (up to 0.85);
`wd=0` wins everywhere, same as the fingerprint MLP.

Net: on this data, **Morgan-Ridge (0.5674) is still the model to beat.** The
from-scratch transformer, the pretrained encoder, and both linear/nonlinear heads
on top of it all confirm the same lesson — engineered fingerprints are hard to top
in a small-data, mostly-linear regime. Beating them likely needs *feature fusion*
(Morgan + ChemBERTa) and *target PCA*, not either representation alone.

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

<details>
<summary><b>Full ChemBERTa→MLP grid sweep</b> (wMSE; best = <b>0.5727</b>)</summary>

MLP on frozen 768-dim ChemBERTa embeddings, same `(lr, epoch, batch, wd)` grid.
Lower is better; the floor is 0.6119 and Morgan-Ridge is 0.5674.

**lr = 1e-3**

| batch | 1k, wd=0 | 1k, wd=1e-4 | 1k, wd=1e-3 | 2.5k, wd=0 | 2.5k, wd=1e-4 | 2.5k, wd=1e-3 |
|---|---|---|---|---|---|---|
| 128 | 0.5755 | 0.6576 | 0.6333 | 0.5903 | 0.5979 | 0.6526 |
| 256 | **0.5727** | 0.6057 | 0.6548 | 0.5882 | 0.6013 | 0.6454 |
| 512 | 0.5791 | 0.5997 | 0.6267 | 0.5812 | 0.5990 | 0.7926 |

**lr = 3e-3**

| batch | 1k, wd=0 | 1k, wd=1e-4 | 1k, wd=1e-3 | 2.5k, wd=0 | 2.5k, wd=1e-4 | 2.5k, wd=1e-3 |
|---|---|---|---|---|---|---|
| 128 | 0.5871 | 0.5927 | 0.6259 | 0.5797 | 0.6046 | 0.7973 |
| 256 | 0.6030 | 0.6105 | 0.8468 | 0.5841 | 0.6017 | 0.6686 |
| 512 | 0.5819 | 0.7144 | 0.6836 | 0.5828 | 0.6024 | 0.6798 |

**lr = 5e-3**

| batch | 1k, wd=0 | 1k, wd=1e-4 | 1k, wd=1e-3 | 2.5k, wd=0 | 2.5k, wd=1e-4 | 2.5k, wd=1e-3 |
|---|---|---|---|---|---|---|
| 128 | 0.5870 | 0.6056 | 0.6668 | 0.5861 | 0.6320 | 0.6549 |
| 256 | 0.5861 | 0.6466 | 0.6438 | 0.5999 | 0.6012 | 0.6812 |
| 512 | 0.5916 | 0.6178 | 0.7697 | 0.5947 | 0.6273 | 0.6412 |

Same story as the fingerprint MLP: `wd=0` wins almost everywhere and `wd=1e-3` is
worst in every block (up to 0.85/0.80). `lr=1e-3` with a small/mid batch and no
weight decay is the sweet spot — best **0.5727** at `lr=1e-3, batch=256, wd=0`.

</details>

## Repo layout

```
vcpi-ml/
├── pyproject.toml          # deps + package metadata (src layout)
├── src/vcpi_ml/
│   ├── sqlshell.py         # remote SQL REPL for exploration         ✅
│   ├── download.py         # data acquisition → data/raw/            ✅
│   ├── data.py             # loaders + split + tokenizer + feature splits ✅
│   ├── expression.py       # counts → log2(CPM+1) target Y           ✅
│   ├── evaluation.py       # wide→long reshape + wMSE scorer wrapper  ✅
│   ├── features.py         # SMILES → Morgan fingerprint / ChemBERTa X ✅
│   ├── device.py           # shared torch device (cuda→mps→cpu)       ✅
│   ├── models/
│   │   ├── mean.py         # per-gene-mean baseline (fit/predict)     ✅
│   │   ├── ridge.py        # Morgan fingerprint → Ridge (fit/predict) ✅
│   │   ├── mlp.py          # PyTorch MLP (fit/predict + train loop)   ✅
│   │   └── smiles_transformer.py  # transformer + fit/predict wrapper      ✅
│   └── experiments/
│       ├── baseline.py     # driver: per-gene-mean baseline           ✅
│       ├── ridge.py        # driver: Ridge (+ alpha sweep)            ✅
│       ├── mlp.py          # driver: MLP (+ hyperparameter grid)      ✅
│       ├── smiles_transformer.py  # driver: transformer (coord. search) ✅
│       ├── chemberta_ridge.py     # driver: ChemBERTa → Ridge          ✅
│       └── chemberta_mlp.py       # driver: ChemBERTa → MLP            ✅
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
6. ✅ Char-level SMILES transformer, hand-written from scratch (multi-head
   attention, Add & Norm encoder blocks, padding mask + masked mean-pool, fit/
   predict wrapper, coordinate-search driver) → **0.5890** tuned. Clears the
   floor but loses to Ridge/MLP — the signal is largely linear and 13k SMILES
   is too little to learn a better representation than a Morgan fingerprint.
   Capacity kept helping (underfitting, not overfitting) → motivates rung 7.
7. ✅ ChemBERTa as a frozen feature extractor → Ridge **0.5794** / MLP **0.5727**.
   A pretrained SMILES encoder (mean-pooled 768-dim embeddings) beats the
   from-scratch transformer, and a nonlinear MLP head beats Ridge on it (the
   dense embedding is less linearly separable than fingerprint bits) — but
   both **still lose to Morgan-Ridge (0.5674)**. Frozen + mean-pool + domain
   mismatch cap the gain; engineered fingerprints stay hard to top. Next
   (optional): *fuse* Morgan + ChemBERTa and PCA the target, the AiBio approach.

Scoring uses the contest package's `score_compounds` / `load_gene_filter` /
`load_weights_matrix` (wrapped in `evaluation.py`); submissions are a parquet of
`(compound, gene_id, predicted_expression)` rows.

## Where it stands

All seven rungs are built, run, and scored — from a per-gene-mean baseline up to
hand-written multi-head attention and a pretrained ChemBERTa encoder. The
consistent finding across four learned models:

| approach | wMSE | vs the 0.6119 floor |
|---|---|---|
| **Ridge on Morgan fingerprints** | **0.5674** | **best** |
| MLP on Morgan fingerprints | 0.5685 | ties Ridge |
| MLP on frozen ChemBERTa | 0.5727 | clears floor, just below best |
| Ridge on frozen ChemBERTa | 0.5794 | clears floor, below Ridge |
| SMILES transformer (from scratch) | 0.5890 | clears floor, below Ridge |

**Morgan-Ridge is the model to beat, and nothing here beats it.** Added nonlinear
capacity (MLP), a from-scratch transformer, and a transformer pretrained on
millions of molecules all clear the floor but land *above* 0.5674 — because the
structure→expression signal is largely linear and a Morgan fingerprint is already
a near-ideal linear basis in this small-data (~13k compound) regime. That's the
project's thesis: **on this data, an engineered fingerprint + linear model is hard
to top; capacity and pretraining only pay off with more data or by *fusing*
representations** (Morgan + ChemBERTa) and compressing the target (PCA) — the
direction a winning pipeline (`AiBio/`) takes.
