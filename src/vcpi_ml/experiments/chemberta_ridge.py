

import numpy as np
import pandas as pd
import torch
from vcpi_prediction_contest import load_gene_filter
from vcpi_ml.data import load_chemberta_split, load_weights
from vcpi_ml.device import DEVICE
from vcpi_ml.experiments.ridge import pipeline

def main():

    genes = set(load_gene_filter())
    X_train, Y_train, X_val, Y_val = load_chemberta_split(
        genes=genes
    )
    weights = load_weights()

    for alpha in [0.1, 1, 10, 100]:
        pipeline(genes, weights, X_train, Y_train, X_val, Y_val, alpha=alpha)



if __name__ == "__main__":
    main()