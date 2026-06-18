

import pandas as pd
import numpy as np

class PerGeneMean:

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.mean = Y.mean(axis=0)
        return self

    def predict(self, X: pd.DataFrame):
        return pd.DataFrame(
                np.tile(self.mean.values, (len(X), 1)),   # (N × genes), every row = mean
                index=X,                                   # val compound IDs
                columns=self.mean.index,                   # gene_ids
            )