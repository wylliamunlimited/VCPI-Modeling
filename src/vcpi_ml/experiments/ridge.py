"""ridge.py — Morgan fingerprint → Ridge regression, scored vs the 0.6119 floor.

First chemistry-aware model: builds fingerprints (X) and expression (Y) via
load_fingerprint_split, fits multi-output Ridge, and scores on the same seed-0
split as the baseline. Beating 0.6119 means the chemistry carries signal.

    uv run python src/vcpi_ml/experiments/ridge.py
"""

import pandas as pd
from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import load_fingerprint_split, load_weights
from vcpi_ml.evaluation import evaluate, wide_to_long
from vcpi_ml.models.ridge import RidgeModel


def main():
    """Build fingerprints + expression, fit Ridge on train, score on val, print wMSE."""

    print("==== Loading in data (w/ train-test-split) ====")
    genes = set(load_gene_filter())
    X_train, Y_train, X_val, Y_val = load_fingerprint_split(
        genes=genes, n_bits=2048, radius=2, n_val=200, seed=0
    )

    print("==== Loading in Weights ====")
    weights = load_weights()

    model = RidgeModel(alpha=1)
    print("==== Fitting Model ====")
    model.fit(X_train, Y_train)
    pred = model.predict(X_val)
    pred = pd.DataFrame(pred, index=Y_val.index, columns=Y_train.columns)

    print("==== Evaluating Model ====")
    ## Reshaping
    pred, truth = (
        wide_to_long(pred, value_col="predicted_expression"),
        wide_to_long(Y_val, value_col="expression"),
    )
    score = evaluate(truth, pred, genes=list(genes), weights=weights)
    print(f"Prediction Score: {score}")


if __name__ == "__main__":
    main()
