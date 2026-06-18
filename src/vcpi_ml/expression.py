"""expression.py — build the contest target Y from raw UMI counts.

Turns raw counts (genes x samples) into the canonical target:
mean-over-replicates of log2(CPM + 1), as a (compounds x genes) matrix.

    CPM   = count / total_umi_count * 1e6      # per-sample sequencing-depth norm
    expr  = log2(CPM + 1)                        # squash scale, keep zeros at 0
    Y     = mean(expr over a compound's replicate samples)

Key correctness point: the CPM denominator is the *whole sample's* depth
(`total_umi_count` from metadata), NOT the column sum of `counts`. Because we
read depth from metadata, it stays correct even after filtering to a gene
subset (`genes`) — which is what makes scoring on the 12,995 graded genes safe.

Verified to match the official `vcpi_prediction_contest.counts_to_expression`
to 0.0 (max abs diff over a 5-compound x 12,995-gene slice).

This is a library module, not a script — import and call it:

    from vcpi_ml.data import load_metadata, load_counts
    from vcpi_ml.expression import counts_to_expression
    Y = counts_to_expression(counts, metadata, genes=set(load_gene_filter()))
"""

import pandas as pd
import numpy as np


def counts_to_expression(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: set[str] | None = None
) -> pd.DataFrame:

    if genes:
        counts = counts.loc[counts.index.isin(genes)] ## [gene_id, *sample_id]

    meta = metadata.assign(sid=metadata["sequenced_id"].astype(str)).set_index("sid")
    depth = meta.loc[counts.columns, "total_umi_count"].to_numpy() ## filtered by gene_id
    compound = meta.loc[counts.columns, "user_compound_id"].to_numpy() ## filtered by gene_id

    c = counts.to_numpy(dtype=np.float64) 
    expr = np.log2(c / depth * 1e6 + 1.0)

    df = pd.DataFrame(expr.T, index=compound, columns=counts.index)

    return df.groupby(level=0).mean()