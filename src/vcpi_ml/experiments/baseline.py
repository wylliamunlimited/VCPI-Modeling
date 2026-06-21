"""baseline.py — run the per-gene-mean baseline end to end and print its wMSE.

The driver that composes every layer (data → expression → model → evaluation)
into one runnable experiment. Establishes the floor every chemistry-aware model
must beat: on the seed-0, 200-compound split it scores ~0.6119 wMSE (verified
== the contest's own predict_per_gene_mean to 0.0).

Note: 0.6119 is this split's number, not the README's 0.507 — the contest never
published its split, so 0.6119 (your fixed split) is the bar to compare against.

    uv run python src/vcpi_ml/experiments/baseline.py
"""

from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import load_expression_split, load_weights
from vcpi_ml.evaluation import evaluate, wide_to_long
from vcpi_ml.models.mean import PerGeneMean


def main():
    """Build expression, fit per-gene mean on train, score on val, print wMSE."""

    print("==== Loading in data (w/ train-test-split) ====")
    genes = set(load_gene_filter())
    train_expression, val_expression = load_expression_split(
        genes=genes, n_val=200, seed=0
    )

    print("==== Loading in Weights ====")
    weights = load_weights()

    model = PerGeneMean()
    print("==== Fitting Model ====")
    model.fit(None, train_expression)
    pred = model.predict(val_expression.index)

    print("==== Evaluating Model ====")
    ## Reshaping
    pred, truth = (
        wide_to_long(pred, value_col="predicted_expression"),
        wide_to_long(val_expression, value_col="expression"),
    )
    score = evaluate(truth, pred, genes=list(genes), weights=weights)
    print(f"Prediction Score: {score}")


if __name__ == "__main__":
    main()
