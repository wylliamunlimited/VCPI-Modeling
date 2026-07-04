"""ridge.py — Morgan fingerprint → Ridge regression, scored vs the 0.6119 floor.

First chemistry-aware model: builds fingerprints (X) and expression (Y) via
load_fingerprint_split, fits multi-output Ridge, and scores on the same seed-0
split as the baseline. Beating 0.6119 means the chemistry carries signal.

    uv run python src/vcpi_ml/experiments/ridge.py
"""

import numpy as np
import pandas as pd
from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import load_fingerprint_split, load_weights
from vcpi_ml.evaluation import evaluate, wide_to_long
from vcpi_ml.models.ridge import RidgeModel


def pipeline(
    genes: set[str],
    weights: pd.DataFrame,
    X_train: pd.DataFrame | np.ndarray,
    Y_train: pd.DataFrame | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    Y_val: pd.DataFrame | np.ndarray,
    alpha: int = 1,
):

    print(f"==== Fitting Model (alpha = {alpha}) ====")
    model = RidgeModel(alpha=alpha).fit(X_train, Y_train)
    pred = model.predict(X_val)
    pred = pd.DataFrame(pred, index=Y_val.index, columns=Y_train.columns)

    print(f"==== Evaluating Model (alpha = {alpha}) ====")
    pred, truth = (
        wide_to_long(pred, value_col="predicted_expression"),
        wide_to_long(Y_val, value_col="expression"),
    )
    score = evaluate(truth, pred, genes=list(genes), weights=weights)
    print(f"Prediction Score (alpha = {alpha}): {score}")


def main():
    """Build fingerprints + expression, fit Ridge on train, score on val, print wMSE."""

    print("==== Loading in data (w/ train-test-split) ====")
    genes = set(load_gene_filter())
    X_train, Y_train, X_val, Y_val = load_fingerprint_split(
        genes=genes, n_bits=2048, radius=2, n_val=200, seed=0
    )

    print("==== Loading in Weights ====")
    weights = load_weights()

    for alpha in [0.1, 1, 10, 100]:
        pipeline(genes, weights, X_train, Y_train, X_val, Y_val, alpha=alpha)


if __name__ == "__main__":
    main()
