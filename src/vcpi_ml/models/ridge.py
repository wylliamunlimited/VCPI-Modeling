"""ridge.py — the first chemistry-aware model: Morgan fingerprint → Ridge.

Maps a molecule's substructure fingerprint (X, from features.morgan_matrix) to
its per-gene expression (Y) with L2-penalized linear regression. sklearn's
Ridge is multi-output, so one fit predicts all ~13k genes at once; the L2
penalty (alpha) keeps the ~2048 correlated fingerprint bits from overfitting.

Same fit/predict contract as PerGeneMean so the driver can swap models in one
line. Following the "DataFrames at the boundary, arrays in the model" rule,
predict returns a raw (n x genes) array — the driver re-attaches the compound
index and gene columns before scoring.

    model = RidgeModel(alpha=1.0).fit(X_train, Y_train)
    preds = model.predict(X_val)          # ndarray; driver labels it
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

class RidgeModel:
    """Multi-output Ridge over Morgan fingerprints (the rung-3 model)."""

    def __init__(self, alpha: float = 1.0):
        """alpha is the L2 strength: higher = more shrinkage = simpler model."""
        self.model: Ridge = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """Fit fingerprints X (n x n_bits) → expression Y (n x genes).

        Rows of X and Y must line up by compound. Returns self for chaining.
        """
        self.model.fit(
            X=X, ## Morgan Fingerprint
            y=Y ## Gene Expression
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expression for fingerprints X → (n x genes) ndarray.

        Unlabeled on purpose; the driver wraps it into a wide DataFrame
        (index=compound ids, columns=gene order) for scoring.
        """
        if not hasattr(self.model, "coef_"):
            raise ValueError("Model not trained - call RidgeModel.fit() before predict()")

        return self.model.predict(
            X=X
        )