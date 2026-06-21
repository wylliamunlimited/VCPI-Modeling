"""mean.py — the per-gene-mean baseline (the floor every real model must beat).

This is NOT a learning model: it ignores the compound entirely (no SMILES, no
chemistry). It just memorizes the average expression of each gene across the
training compounds and predicts that same vector for everyone. It answers
"how well can you do knowing nothing about the molecule?" — the answer is
~0.5 wMSE, the bar any chemistry-aware model has to clear to justify itself.

It implements the shared fit/predict contract so the driver can swap it for a
real model (Ridge, MLP, attention) with a one-line change.

    model = PerGeneMean().fit(None, Y_train)   # X ignored
    Y_pred = model.predict(val_ids)            # every row identical
"""

import pandas as pd
import numpy as np


class PerGeneMean:
    """Predicts the train-set per-gene mean for every compound, ignoring X."""

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """Store the per-gene mean over training compounds.

        X is ignored (kept for interface uniformity). Y is the wide train
        expression matrix (compounds x genes); axis=0 averages down the
        compounds → one value per gene.
        """
        self.mean = Y.mean(axis=0)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Broadcast the stored mean to one identical row per compound in X.

        X is the compound IDs to predict for; returns a wide
        (len(X) x genes) matrix indexed by those IDs.
        """
        return pd.DataFrame(
                np.tile(self.mean.values, (len(X), 1)),   # (N × genes), every row = mean
                index=X,                                   # val compound IDs
                columns=self.mean.index,                   # gene_ids
            )