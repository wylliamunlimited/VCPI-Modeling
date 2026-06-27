
from vcpi_prediction_contest import (
    load_gene_filter
)
from vcpi_ml.data import (
    load_fingerprint_split, load_weights
)
from vcpi_ml.evaluation import evaluate, wide_to_long
from vcpi_ml.models.mlp import MLPModel, DEVICE
import pandas as pd
import numpy as np
import torch

GRID = {
    "lr":           [1e-3, 3e-3, 5e-3],
    "epoch":        [1000, 2500],
    "batch":        [128, 256, 512],
    "weight_decay": [0.0, 1e-4, 1e-3],
}

def pipeline(
    genes: set[str], weights: pd.DataFrame,
    X_train: torch.Tensor, Y_train: torch.Tensor,
    X_val: pd.DataFrame | np.ndarray, Y_val: pd.DataFrame | np.ndarray,
    gene_cols: pd.Index, lr: float = 0.01,
    batch_size: int = 256, epoch: int = 2048, weight_decay: float = 0.0
    ):

    params = f"lr={lr}, epoch={epoch}, batch={batch_size}, wd={weight_decay}"
    print(f"===== Training MLP ({params}) =====")
    model = MLPModel(lr=lr, weight_decay=weight_decay)
    model.fit(X=X_train, Y=Y_train, epoch=epoch, batch=batch_size)

    pred = model.predict(X=X_val)
    pred = pd.DataFrame(pred, index=Y_val.index, columns=gene_cols)
    pred, truth = (
        wide_to_long(pred, value_col="predicted_expression"),
        wide_to_long(Y_val, value_col="expression"),
    )
    score = evaluate(truth, pred, genes=list(genes), weights=weights)
    print(f"\t{params} -> wMSE={score:.4f} (train MSE={model.history[-1]:.4f})")
    print()


def main():
    print("==== Loading in data (w/ train-test-split) ====")
    genes = set(load_gene_filter())
    X_train, Y_train, X_val, Y_val = load_fingerprint_split(
        genes=genes, n_bits=2048, radius=2, n_val=200, seed=0
    )

    print("==== Loading in Weights ====")
    weights = load_weights()

    # Convert the train arrays to device tensors once, not per grid run.
    gene_cols = Y_train.columns
    X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.float32, device=DEVICE)

    for _lr in GRID.get("lr", []):
        for ep in GRID.get("epoch", []):
            for _batch_size in GRID.get("batch", []):
                for _wd in GRID.get("weight_decay", []):
                    pipeline(
                        genes, weights, X_train, Y_train, X_val, Y_val, gene_cols,
                        lr=_lr, batch_size=_batch_size, epoch=ep, weight_decay=_wd
                        )
    

if __name__ == "__main__":
    main()