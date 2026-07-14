"""chemberta_ridge.py — driver: transfer learning, ChemBERTa → Ridge (rung 7).

The controlled transfer-learning experiment: swap Morgan fingerprints for frozen
pretrained ChemBERTa embeddings, keep the *exact same* Ridge on top (reuses
ridge.py's pipeline), and see if a SMILES encoder pretrained on ~millions of
molecules beats hand-designed features. Only the featurizer differs from the
Morgan-Ridge run, so any score gap is purely representation.

Result: best 0.5794 (α=100) — clears the floor and beats the from-scratch
transformer, but still loses to Morgan-Ridge's 0.5674 (the task is linear and
Morgan is already a near-ideal linear basis).

    uv run python src/vcpi_ml/experiments/chemberta_ridge.py
"""

import numpy as np
import pandas as pd
import torch
from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import load_chemberta_split, load_weights
from vcpi_ml.device import DEVICE
from vcpi_ml.experiments.ridge import pipeline


def main():
    """Embed all SMILES with frozen ChemBERTa once, then sweep Ridge alpha."""
    genes = set(load_gene_filter())
    X_train, Y_train, X_val, Y_val = load_chemberta_split(
        genes=genes
    )
    weights = load_weights()

    for alpha in [0.1, 1, 10, 100]:
        pipeline(genes, weights, X_train, Y_train, X_val, Y_val, alpha=alpha)



if __name__ == "__main__":
    main()