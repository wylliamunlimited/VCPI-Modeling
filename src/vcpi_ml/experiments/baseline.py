"""baseline.py — run the per-gene-mean baseline end to end and print its wMSE.

The driver that composes every layer (data → expression → model → evaluation)
into one runnable experiment. Establishes the floor every chemistry-aware model
must beat: on the seed-0, 200-compound split it scores ~0.6119 wMSE (verified
== the contest's own predict_per_gene_mean to 0.0).

Note: 0.6119 is this split's number, not the README's 0.507 — the contest never
published its split, so 0.6119 (your fixed split) is the bar to compare against.

    uv run python src/vcpi_ml/experiments/baseline.py
"""

import pandas as pd
from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import (
    load_metadata, load_counts, load_weights, split_compounds
)
from vcpi_ml.evaluation import evaluate, wide_to_long
from vcpi_ml.expression import counts_to_expression
from vcpi_ml.models.mean import PerGeneMean

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load + split, returning train/test (metadata, counts) for each side.

    Splits on compound, translates each side's compounds to their sample ids,
    then reads only those count columns. Returns the metadata subset matching
    each side's samples, so counts_to_expression can align depth/compound.
    """

    metadata: pd.DataFrame = load_metadata() # Loaded dataset is already filtered by THP-1 & 24hr timepoint
    train_compounds, test_compounds = split_compounds(metadata=metadata)
    train_sample_ids, test_sample_ids = (
        metadata.loc[metadata["user_compound_id"].isin(train_compounds), "sequenced_id"],
        metadata.loc[metadata["user_compound_id"].isin(test_compounds), "sequenced_id"],
    )
    train_counts, test_counts = (
        load_counts(train_sample_ids.astype(str).tolist()), load_counts(test_sample_ids.astype(str).tolist())
    )
    train_metadata, test_metadata = (
        metadata.loc[metadata["sequenced_id"].isin(train_sample_ids)],
        metadata.loc[metadata["sequenced_id"].isin(test_sample_ids)]
    )
    return train_metadata, train_counts, test_metadata, test_counts


def main():
    """Build expression, fit per-gene mean on train, score on val, print wMSE."""

    print(f"==== Loading in data (w/ train-test-split) ====")
    train_metadata, train_counts, test_metadata, test_counts = load_data()
    genes = set(load_gene_filter())
    train_expression, test_expression = (
        counts_to_expression(train_counts, train_metadata, genes=genes), 
        counts_to_expression(test_counts, test_metadata, genes=genes)
    )

    print(f"==== Loading in Weights ====")
    weights = load_weights()

    model = PerGeneMean()
    print(f"==== Fitting Model ====")
    model.fit(
        None, train_expression
    )
    pred = model.predict(
        test_expression.index
    )
    
    print(f"==== Evaluating Model ====")
    ## Reshaping
    pred, truth = (
        wide_to_long(pred, value_col="predicted_expression"), 
        wide_to_long(test_expression, value_col="expression")
    )
    score = evaluate(truth, pred, genes=list(genes), weights=weights)
    print(f"Prediction Score: {score}")


if __name__ == "__main__":
    main()