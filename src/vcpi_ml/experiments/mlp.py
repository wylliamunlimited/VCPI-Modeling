
from vcpi_prediction_contest import (
    load_gene_filter
)
from vcpi_ml.data import (
    load_fingerprint_split, load_weights
)
from vcpi_ml.evaluation import evaluate, wide_to_long
from vcpi_ml.models.mlp import MLPModel
import pandas as pd
import numpy as np

def pipeline(
    genes: set[str], weights: pd.DataFrame, 
    X_train: pd.DataFrame | np.ndarray, Y_train: pd.DataFrame | np.ndarray, 
    X_val: pd.DataFrame | np.ndarray, Y_val: pd.DataFrame | np.ndarray, lr: float = 0.01
    ):

    print(f"===== Initializing MLP Model (lr = {lr}) ======")
    model = MLPModel(lr=lr)
    model.fit(X=X_train, Y=Y_train)

    print(f"===== Evaluating Model (lr = {lr}) =====")
    pred = model.predict(X=X_val)
    pred = pd.DataFrame(pred, index=Y_val.index, columns=Y_train.columns)
    pred, truth = (
        wide_to_long(pred, value_col="predicted_expression"),
        wide_to_long(Y_val, value_col="expression"),
    )
    score = evaluate(truth, pred, genes=list(genes), weights=weights)
    print(f"\tPrediction Score (lr = {lr}): {score}")
    print()


def main():
    print("==== Loading in data (w/ train-test-split) ====")
    genes = set(load_gene_filter())
    X_train, Y_train, X_val, Y_val = load_fingerprint_split(
        genes=genes, n_bits=2048, radius=2, n_val=200, seed=0
    )

    print("==== Loading in Weights ====")
    weights = load_weights()

    lr = [1e-5, 1e-3, 0.1]

    for _lr in lr:
        pipeline(genes, weights, X_train, Y_train, X_val, Y_val, _lr)
        
    

if __name__ == "__main__":
    main()