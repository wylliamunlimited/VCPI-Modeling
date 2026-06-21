"""data.py — read-side of the pipeline: load what download.py saved, split it.

download.py pulls the contest data from VCPI to data/raw/ (run once); this
module reads those parquet files back into memory for every experiment. The
split is on *compounds*, never samples, so a compound's replicates never
straddle the train/val boundary (that would leak the answer).

    metadata  — one row per sample (the join key + total_umi_count for the target)
    counts    — genes x samples, raw UMI counts (the un-normalized target)

Note: these parquet files are pyarrow-backed, so columns read back as Arrow
arrays. Convert to NumPy at the boundary (see split_compounds) before handing
them to libraries like sklearn that expect integer-indexable arrays.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[2] / "data" / "raw"

def load_metadata() -> pd.DataFrame:
    """One row per sample. Read train_metadata.parquet."""
    return pd.read_parquet(DATA / "train_metadata.parquet")

def load_counts(sample_ids: list[str]) -> pd.DataFrame:
    """Gene x samples. Read ONLY the columns we need:
        pd.read_parquet(..., columns=["gene_id", *sample_ids]).
        Return with gene_id as the index."""
    return pd.read_parquet(DATA / "train_counts.parquet", columns=["gene_id", *sample_ids]).set_index("gene_id")

def split_compounds(metadata, n_val=200, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Unique non-control user_compound_ids -> shuffle(seed) -> (train, val).
        Return compound IDs, never sample IDs."""
    compounds = np.asarray(metadata.loc[~metadata["is_control"], "user_compound_id"].unique())
    train_data, test_data = train_test_split(compounds, test_size=n_val, random_state=seed)
    return train_data, test_data

def load_weights() -> pd.DataFrame:
    """Mejia scoring weights: (genes x compounds), gene_id index.

    Reads the local cache download.py saved (== the contest's
    load_weights_matrix() output), so no re-fetch is needed. Each compound
    column sums to ~1.0; pass this to evaluation.evaluate() as the weights
    that make the wMSE match the leaderboard.
    """
    return pd.read_parquet(DATA / "weights.parquet")