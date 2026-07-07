from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import load_token_split, load_weights
from vcpi_ml.device import DEVICE
from vcpi_ml.evaluation import evaluate, wide_to_long
from vcpi_ml.models.smiles_transformer import TransformerModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Modest sweep — coordinate search, NOT a full cartesian product (the transformer
# is ~2M params and slow to train). Tune top-to-bottom: fix each knob at its best
# before moving to the next. n_heads stays fixed so d_k = d_model / n_heads stays
# in the healthy 32-64 range.
GRID = {
    "lr": [3e-4, 1e-3, 3e-3],  # 1st: matters most for whether it trains
    "n_attention": [1, 2, 4],  # 2nd: depth — the main capacity dial
    "d_model": [128, 256],  # 3rd: width — params scale fast
    "dropout": [0.0, 0.1],  # 4th: regularizer (weight decay hurt the MLP)
    "batch": [32, 64],  # mostly speed / stability
    "n_heads": [4],  # fixed; keep d_model % n_heads == 0
    "epochs": [50],  # ceiling; early stopping ends most runs sooner
}

MAX_SEQ_LEN = 128


def pipeline(
    genes: set[str],
    weights: pd.DataFrame,
    vocabs: dict[str, int],
    X_train: pd.DataFrame | np.ndarray,  # (batch, seq_len)
    Y_train: pd.DataFrame | np.ndarray,  # (batch, seq_len)
    X_val: pd.DataFrame | np.ndarray,  # (batch, n_genes)
    Y_val: pd.DataFrame | np.ndarray,  # (batch, n_genes)
    mask_train: pd.DataFrame | np.ndarray,
    mask_val: pd.DataFrame | np.ndarray,
    gene_cols: pd.Index,
    n_heads: int,
    n_attention: int,
    d_model: int,
    dropout: float,
    epoch: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0
):

    params = (
        f"lr={lr}, epoch={epoch}, batch={batch_size}, d_model={d_model}, "
        f"n_attention={n_attention}, heads={n_heads}, dropout={dropout}"
    )
    print(f"===== Training Transformer ({params}) =====")

    model = TransformerModel(
        vocab_size=len(vocabs),
        n_genes=len(genes), d_model=d_model,
        n_heads=n_heads, n_attention=n_attention,
        max_len=MAX_SEQ_LEN, dropout=dropout,
        lr=lr, weight_decay=weight_decay
    )

    model.fit(X_train, Y_train, mask_train, epoch, batch_size)

    pred = model.predict(X_val, mask_val)
    pred = pd.DataFrame(pred, index=Y_val.index, columns=gene_cols)

    pred, truth = (
        wide_to_long(pred, value_col="predicted_expression"),
        wide_to_long(Y_val, value_col="expression"),
    )

    score = evaluate(truth, pred, list(genes), weights)
    print(f"\t{params} -> wMSE={score:.4f} (train MSE={model.history[-1]:.4f})")
    print()
    return score


def main():
    """Load once, then coordinate-search the GRID (one knob at a time).

    A full cartesian product would be 3*3*2*2*2 = 72 slow runs. Instead we sweep
    knobs in priority order (lr, then depth, width, dropout, batch), fixing each
    at its best-scoring value before moving to the next — ~12 runs total. Not
    guaranteed globally optimal, but a sane budget for a from-scratch model.
    """
    # Load data once (numpy/DataFrames; the wrapper converts to tensors per run).
    genes = set(load_gene_filter())
    X_train, mask_train, Y_train, X_val, mask_val, Y_val, vocabs = load_token_split(
        genes, max_len=MAX_SEQ_LEN, n_val=200
    )
    weights = load_weights()
    gene_cols = Y_train.columns  # capture gene ids before Y_train is used as data

    # Fixed args every pipeline() call needs (the data + labels).
    data = dict(
        genes=genes, weights=weights, vocabs=vocabs,
        X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
        mask_train=mask_train, mask_val=mask_val, gene_cols=gene_cols,
    )

    # Starting point for the search — every knob at a sensible default.
    best = dict(
        lr=1e-3, n_attention=2, d_model=128, dropout=0.1,
        batch_size=64, n_heads=4, epoch=50,
    )

    # (GRID key -> config key) in priority order; GRID's "batch"/"epochs" names
    # differ from pipeline's "batch_size"/"epoch", so map them explicitly.
    sweep = [
        ("lr", "lr"),
        ("n_attention", "n_attention"),
        ("d_model", "d_model"),
        ("dropout", "dropout"),
        ("batch", "batch_size"),
    ]

    best_score = float("inf")
    for grid_key, cfg_key in sweep:
        knob_best_val, knob_best_score = best[cfg_key], best_score
        for val in GRID[grid_key]:
            score = pipeline(**data, **{**best, cfg_key: val})
            if score < knob_best_score:
                knob_best_score, knob_best_val = score, val
        best[cfg_key], best_score = knob_best_val, knob_best_score
        print(f">>> fixed {cfg_key}={knob_best_val}  (best wMSE so far {best_score:.4f})\n")

    print(f"===== Best config: {best}  ->  wMSE={best_score:.4f} =====")



if __name__ == "__main__":
    main()
