"""chemberta_mlp.py — driver: transfer learning, ChemBERTa → MLP (rung 7).

Same frozen-ChemBERTa featurizer as chemberta_ridge.py, but with a nonlinear
head: reuses mlp.py's pipeline (the MLP infers n_in from the 768-dim embedding)
and sweeps the same lr × epoch × batch × weight-decay grid. Tests whether a
nonlinear model can pull more out of the pretrained embedding than Ridge did —
though, as with fingerprints, the MLP tends to only tie its linear counterpart
when the signal is largely linear.

Embeddings/train tensors are built once, before the grid, so nothing reloads
per config.

    uv run python src/vcpi_ml/experiments/chemberta_mlp.py
"""

import numpy as np
import pandas as pd
import torch
from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import load_chemberta_split, load_weights
from vcpi_ml.device import DEVICE
from vcpi_ml.experiments.mlp import pipeline

GRID = {
    "lr": [1e-3, 3e-3, 5e-3],
    "epoch": [1000, 2500],
    "batch": [128, 256, 512],
    "weight_decay": [0.0, 1e-4, 1e-3],
}


def main():
    """Embed all SMILES with frozen ChemBERTa once, convert train tensors, sweep the grid."""
    genes = set(load_gene_filter())
    X_train, Y_train, X_val, Y_val = load_chemberta_split(
        genes=genes
    )
    weights = load_weights()

    gene_cols = Y_train.columns
    X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.float32, device=DEVICE)

    for _lr in GRID.get("lr", []):
        for ep in GRID.get("epoch", []):
            for _batch_size in GRID.get("batch", []):
                for _wd in GRID.get("weight_decay", []):
                    pipeline(
                        genes,
                        weights,
                        X_train,
                        Y_train,
                        X_val,
                        Y_val,
                        gene_cols,
                        lr=_lr,
                        batch_size=_batch_size,
                        epoch=ep,
                        weight_decay=_wd
                    )



if __name__ == "__main__":
    main()