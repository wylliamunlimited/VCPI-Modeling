
import pandas as pd
from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import (
    load_metadata, load_counts, load_weights, split_compounds
)
from vcpi_ml.evaluation import evaluate, wide_to_long
from vcpi_ml.expression import counts_to_expression
from vcpi_ml.models.mean import PerGeneMean

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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