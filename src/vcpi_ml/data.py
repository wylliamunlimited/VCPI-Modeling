"""data.py — load what download.py saved and assemble modeling-ready datasets.

download.py pulls the contest data from VCPI to data/raw/ (run once); this
module reads those parquet files back into memory for every experiment. Two
layers:

  - Raw loaders (load_metadata/chemistry/counts/weights) — parquet -> DataFrame.
  - Assembly (split_compounds, load_expression_split, load_fingerprint_split) —
    compose the loaders + expression/features into the (X, Y) an experiment
    trains on, so the drivers stay thin and every model loads data identically.

The split is on *compounds*, never samples, so a compound's replicates never
straddle the train/val boundary (that would leak the answer).

    metadata  — one row per sample (the join key + total_umi_count for the target)
    counts    — genes x samples, raw UMI counts (the un-normalized target)
    chemistry — one row per compound (SMILES -> the model input X)

Note: these parquet files are pyarrow-backed, so columns read back as Arrow
arrays. Convert to NumPy at the boundary (see split_compounds) before handing
them to libraries like sklearn that expect integer-indexable arrays.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from vcpi_ml.expression import counts_to_expression
from vcpi_ml.features import chemberta_matrix, morgan_matrix

DATA = Path(__file__).resolve().parents[2] / "data" / "raw"

# = = = = = RAW DATA LOADING = = = = =


def load_metadata() -> pd.DataFrame:
    """One row per sample. Read train_metadata.parquet."""
    return pd.read_parquet(DATA / "train_metadata.parquet")


def load_chemistry() -> pd.DataFrame:
    """One row per compound: SMILES + RDKit descriptors. Read train_chemistry.parquet.

    The source of model inputs X (the chemistry side); join to metadata on
    `compound` to map user_compound_id -> SMILES.
    """
    return pd.read_parquet(DATA / "train_chemistry.parquet")


def load_counts(sample_ids: list[str]) -> pd.DataFrame:
    """Gene x samples. Read ONLY the columns we need:
    pd.read_parquet(..., columns=["gene_id", *sample_ids]).
    Return with gene_id as the index."""
    return pd.read_parquet(
        DATA / "train_counts.parquet", columns=["gene_id", *sample_ids]
    ).set_index("gene_id")


def load_weights() -> pd.DataFrame:
    """Mejia scoring weights: (genes x compounds), gene_id index.

    Reads the local cache download.py saved (== the contest's
    load_weights_matrix() output), so no re-fetch is needed. Each compound
    column sums to ~1.0; pass this to evaluation.evaluate() as the weights
    that make the wMSE match the leaderboard.
    """
    return pd.read_parquet(DATA / "weights.parquet")


# = = = = = Processing = = = = =


def split_compounds(metadata, n_val=200, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Unique non-control user_compound_ids -> shuffle(seed) -> (train, val).
    Return compound IDs, never sample IDs."""
    compounds = np.asarray(
        metadata.loc[~metadata["is_control"], "user_compound_id"].unique()
    )
    train_data, test_data = train_test_split(
        compounds, test_size=n_val, random_state=seed
    )
    return train_data, test_data


def _expression_for(
    metadata: pd.DataFrame,
    compounds_ids: list | np.ndarray | pd.Series,
    genes: set[str] | None = None,
) -> pd.DataFrame:
    """Build the wide (compounds x genes) expression matrix for given compounds.

    Translates compound ids -> their sample ids, loads only those count columns,
    and runs counts_to_expression. `genes` restricts to the scored gene set.
    """
    sids = (
        metadata.loc[metadata["user_compound_id"].isin(compounds_ids), "sequenced_id"]
        .astype(str)
        .tolist()
    )
    return counts_to_expression(load_counts(sids), metadata=metadata, genes=genes)


def load_expression_split(
    genes: set[str] | None = None, n_val: int = 200, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Y_train, Y_val) expression matrices, split on compound.

    The modeling-ready target for the per-gene-mean baseline: load, split on
    compound (leak-free), and build expression for each side. Each is a wide
    (compounds x genes) DataFrame indexed by user_compound_id.
    """
    metadata: pd.DataFrame = load_metadata()
    train_compounds, test_compounds = split_compounds(metadata, n_val=n_val, seed=seed)
    return (
        _expression_for(metadata=metadata, compounds_ids=train_compounds, genes=genes),
        _expression_for(metadata=metadata, compounds_ids=test_compounds, genes=genes),
    )


def smiles_by_compound() -> pd.Series:
    """user_compound_id -> SMILES, as a Series indexed by compound id."""
    return load_chemistry().set_index("user_compound_id")["smiles"]


def load_fingerprint_split(
    genes: set[str] | None = None,
    n_bits: int = 2048,
    radius: int = 2,
    n_val: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(X_train, Y_train, X_val, Y_val) for the Morgan-fingerprint -> expression task.

    Builds the expression split, then fingerprints each side's compounds in the
    *same row order* as Y (via reindex on Y.index), so X and Y can't desync.
    X is a numpy (compounds x n_bits) matrix; Y stays a labelled DataFrame
    (its index/columns are the compound/gene ids the scorer needs).
    """
    Y_train, Y_val = load_expression_split(genes, n_val, seed)
    smiles = smiles_by_compound()
    X_train = morgan_matrix(smiles.reindex(Y_train.index), n_bits, radius)
    X_val = morgan_matrix(smiles.reindex(Y_val.index), n_bits, radius)
    return X_train, Y_train, X_val, Y_val


def build_smile_char_vocab(
    smiles_list: list | np.ndarray | pd.Series,
) -> dict[str, int]:
    """Build a char -> id vocabulary for SMILES (char-level tokenization).

    Ids start at 1; **id 0 is reserved for <pad>** so padding has its own id the
    embedding and mask agree on. Fit this on TRAIN smiles only and reuse it for
    val (a shared vocabulary — never let val invent its own ids).

    Note: char-level splits two-character atoms (Cl, Br) and bracket atoms
    ([nH], [O-]) into separate chars — chemically imprecise but workable; swap
    for a regex SMILES tokenizer if it limits the model.
    """
    chars = sorted({c for s in smiles_list for c in s})
    return {c: i + 1 for i, c in enumerate(chars)}


def tokenize_smile_char(
    smiles_list: list | np.ndarray | pd.Series, vocab: dict[str, int], max_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Char-level encode + pad SMILES to max_len.

    Returns (tokens, mask), both (n, max_len) int64 arrays:
      tokens[i] = char ids of smiles i, right-padded with 0 (truncated at max_len)
      mask[i]   = 1 for real tokens, 0 for padding (so attention/pooling skip pad)
    Unknown chars (unseen in train) fall back to 0; int64 feeds nn.Embedding (Long).
    """
    n = len(smiles_list)  # Sample count
    tokens = np.zeros((n, max_len), dtype=np.int64)
    mask = np.zeros((n, max_len), dtype=np.int64)

    for i, s in enumerate(smiles_list):
        ids = [vocab.get(c, 0) for c in s][:max_len]
        tokens[i, : len(ids)] = ids
        mask[i, : len(ids)] = 1

    return tokens, mask


def load_token_split(
    genes: set[str] | None = None,
    max_len: int = 128,
    n_val: int = 200,
    seed: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    dict[str, int],
]:
    """Token dataset for the SMILES transformer (rung 6).

    Builds the expression split, then char-tokenizes each side's SMILES in the
    *same row order* as Y (reindex on Y.index, so X and Y can't desync). The
    vocab is fit on TRAIN only and reused for val.

    Returns (X_train, mask_train, Y_train, X_val, mask_val, Y_val, vocab):
      X_*    — (compounds x max_len) int64 token ids
      mask_* — (compounds x max_len) 1=real / 0=pad
      Y_*    — labelled expression DataFrames (compound/gene ids for scoring)
      vocab  — char->id map; len(vocab)+1 is the embedding vocab_size (+1 for pad)
    """
    Y_train, Y_val = load_expression_split(genes, n_val, seed)
    smiles = smiles_by_compound()
    train_smiles, val_smiles = (
        smiles.reindex(Y_train.index),
        smiles.reindex(Y_val.index),
    )

    vocabs = build_smile_char_vocab(train_smiles)
    X_train, mask_train = tokenize_smile_char(train_smiles, vocabs, max_len)
    X_val, mask_val = tokenize_smile_char(val_smiles, vocabs, max_len)
    return X_train, mask_train, Y_train, X_val, mask_val, Y_val, vocabs


def load_chemberta_split(
    genes: set[str] | None = None,
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 64,
    max_len: int = 128,
    n_val: int = 200,
    seed: int = 0
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(X_train, Y_train, X_val, Y_val) for the ChemBERTa -> expression task (rung 7).

    Identical shape/contract to load_fingerprint_split, but X is a frozen
    pretrained-transformer embedding instead of a Morgan fingerprint — so a
    driver can swap featurizers and compare like-for-like (same Ridge on top).
    Builds the expression split, then embeds each side's SMILES in the *same row
    order* as Y (reindex on Y.index, so X and Y can't desync). X is a numpy
    (compounds x 768) matrix; Y stays a labelled DataFrame for the scorer.

    Note: recomputes the embeddings on every call (no cache) — ~14k SMILES
    through the transformer takes a few minutes. Fine for a one-shot run; add a
    parquet cache here if you iterate on the driver a lot.
    """
    Y_train, Y_val = load_expression_split(genes, n_val, seed)
    smiles = smiles_by_compound()
    train_smiles, val_smiles = (
        smiles.reindex(Y_train.index),
        smiles.reindex(Y_val.index)
    )

    X_train, X_val = (
        chemberta_matrix(train_smiles, model_name, batch_size, max_len),
        chemberta_matrix(val_smiles, model_name, batch_size, max_len)
    )
    return X_train, Y_train, X_val, Y_val