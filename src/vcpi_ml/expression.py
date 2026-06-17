

import pandas as pd
import numpy as np

def counts_to_expression(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: set[str] | None = None
) -> pd.DataFrame:
    
    counts = counts.loc[counts.index.isin(genes)] ## [gene_id, *sample_id]

    meta = metadata.assign(sid=metadata["sequenced_id"].astype(str)).set_index("sid")
    depth = meta.loc[counts.columns, "total_umi_count"].to_numpy()
    compound = meta.loc[counts.columns, "user_compound_id"].to_numpy()

    c = counts.to_numpy(dtype=np.float64) ## cpm * 1M ==> 
    expr = np.log2(c / depth * 1e6 + 1.0)

    df = pd.DataFrame(expr.T, index=compound, columns=counts.index)

    return df.groupby(level=0).mean()