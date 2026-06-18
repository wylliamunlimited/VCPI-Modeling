"""evaluation.py — score predictions with the official contest metric.

A thin adapter over the contest scorer. Our models produce a *wide*
(compounds x genes) expression matrix; the scorer wants *long* rows. This
module reshapes (wide_to_long) and scores (evaluate), so an experiment can
get a single wMSE number in one call without re-deriving the scorer's quirks.

We intentionally do NOT reimplement the metric: score_compounds is the
external judge (it must match the leaderboard exactly), so we wrap it rather
than risk divergence. Always pass the real Mejia weights — without them the
scorer silently falls back to a different weighting and the number is
meaningless vs. the leaderboard.

    from vcpi_prediction_contest import load_gene_filter, load_weights_matrix
    truth = wide_to_long(Y_val,  "expression")
    pred  = wide_to_long(Y_pred, "predicted_expression")
    wmse  = evaluate(truth, pred, list(load_gene_filter()), load_weights_matrix())
"""

import pandas as pd

from vcpi_prediction_contest import (
    score_compounds,
    aggregate_leaderboards
)

def wide_to_long(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Reshape a wide expression matrix into the scorer's long format.

    The compound index becomes a column literally named ``compound`` (the
    name score_compounds hard-codes); the gene columns are unpivoted into a
    ``gene_id`` column; cell values land in ``value_col``. Keys are cast to
    plain str so the scorer's internal alignment doesn't choke on Arrow dtypes.

    Parameters
    ----------
    df
        Wide matrix: index = user_compound_id, columns = gene_id, cells =
        log2(CPM+1) expression.
    value_col
        Name for the value column — ``"expression"`` for truth,
        ``"predicted_expression"`` for predictions.

    Returns
    -------
    Long DataFrame with columns ``compound | gene_id | <value_col>``,
    one row per (compound, gene).
    """

    df = df.reset_index(names="compound")
    df = df.melt(id_vars="compound", var_name="gene_id", value_name=value_col)
    df["compound"], df["gene_id"] = df["compound"].astype(str), df["gene_id"].astype(str)
    return df


def evaluate(truth: pd.DataFrame, pred: pd.DataFrame, genes: list[str], weights: pd.DataFrame) -> float:
    """Score predictions against truth with the contest's weighted MSE.

    Wraps the two contest calls (score_compounds → aggregate_leaderboards)
    into one number. Both frames must be long format (see wide_to_long).

    Parameters
    ----------
    truth
        Long frame: ``compound | gene_id | expression``.
    pred
        Long frame: ``compound | gene_id | predicted_expression``.
    genes
        gene_ids to score on (the contest's load_gene_filter() set).
    weights
        The Mejia (genes x compounds) weight matrix from
        load_weights_matrix(). Required — omitting it would make the scorer
        fall back to a different weighting that doesn't match the leaderboard.

    Returns
    -------
    The mean weighted-MSE across compounds (lower is better).
    """

    per_compound = score_compounds(
        truth, pred,
        gene_filter=genes, weights=weights
    )
    board = aggregate_leaderboards(per_compound=per_compound)
    return board["wmse_mean"]