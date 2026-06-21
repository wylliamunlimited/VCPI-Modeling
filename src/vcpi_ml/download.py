"""download.py — pull the VCPI contest data into data/raw/.

Run with the token loaded from your .env:

    uv run --env-file .env python src/vcpi_ml/download.py

Produces four parquet files in data/raw/:
    weights.parquet          ~358 MB   Mejia scoring weights (genes x compounds)
    train_counts.parquet     ~1.4 GB   raw UMI counts (genes x samples)
    train_metadata.parquet   ~3 MB     one row per sample
    train_chemistry.parquet  ~1.5 MB   one row per compound (SMILES, etc.)

Peak RAM ~22 GB, a few minutes on a fast network.
"""

import os
from pathlib import Path

import pandas as pd
import polars as pl
import vcpi
from vcpi_prediction_contest import load_weights_matrix

# parents[2] = the repo root (this file is at <root>/src/vcpi_ml/download.py),
# then into data/raw/. mkdir so a fresh clone doesn't blow up.
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
JOBS = ["tvc-bhr-009", "tvc-kdl-010", "tvc-qnu-012"]


def parse_exp(exp: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split one experiment dict into its three polars frames."""
    return exp["data"], exp["metadata"], exp["chemistry"]


def filter_to_contest_condition(sample_metadata: pl.DataFrame) -> pl.DataFrame:
    """Keep only THP-1 / 24h / 10 uM library samples + DMSO controls.

    vcpi stores 10 uM as 10000 nM, hence the unit check.
    """
    return sample_metadata.filter(
        (pl.col("cell_line") == "THP-1")
        & (pl.col("timepoint") == "24h")
        & (
            (
                (pl.col("compound_concentration") == 10_000)
                & (pl.col("compound_concentration_unit") == "nM")
            )
            | (pl.col("user_compound_id") == "DMSO")
        )
    )


def select_kept_count_columns(
    gene_expression: pl.DataFrame, sample_metadata: pl.DataFrame
) -> pl.DataFrame:
    """Keep gene_id + only the count columns for samples in the filtered metadata.

    Counts are wide (gene_id + one column per sample), so we subset *columns*
    down to the samples that survived the contest-condition filter.
    """
    keep = set(sample_metadata["sequenced_id"].cast(pl.Utf8).to_list())
    cols = [
        "gene_id",
        *[c for c in gene_expression.columns if c != "gene_id" and c in keep],
    ]
    return gene_expression.select(cols)


def download_weights() -> None:
    """Stream the Mejia scoring weights (public GitHub release, not token-gated)."""
    print("Downloading weights matrix...")
    load_weights_matrix().to_parquet(RAW / "weights.parquet")


def download_counts_metadata_chemistry() -> None:
    """Loop the 3 releases, filter, merge, and write the 3 training parquet files."""
    counts_pieces, metadata_pieces, chemistry_pieces = [], [], []

    for job in JOBS:
        print(f"Loading {job}...")
        gene_expression, sample_metadata, compound_chemistry = parse_exp(
            vcpi.load_experiment(job)
        )

        sample_metadata = filter_to_contest_condition(sample_metadata)
        gene_expression = select_kept_count_columns(gene_expression, sample_metadata)

        # gene_id as the index so the column-wise concat below aligns on genes.
        counts_pieces.append(gene_expression.to_pandas().set_index("gene_id"))
        metadata_pieces.append(sample_metadata.to_pandas())
        chemistry_pieces.append(compound_chemistry.to_pandas())

        # free the big frames before the next iteration (RAM matters here).
        del gene_expression, sample_metadata, compound_chemistry

    print("Merging releases...")
    # Counts: concat along columns — each job adds *samples*, not genes. Genes
    # missing from a job become NaN, so fill with 0 and cast back to int counts.
    unified_gene_expression = (
        pd.concat(counts_pieces, axis=1, join="outer")
        .fillna(0)
        .astype("int32")
        .reset_index()
    )
    # Metadata: stack rows (one row per sample).
    unified_sample_metadata = pd.concat(metadata_pieces, ignore_index=True)
    # Chemistry: stack rows, then drop the compounds that repeat across releases.
    unified_compound_chemistry = (
        pd.concat(chemistry_pieces, ignore_index=True)
        .drop_duplicates(subset=["compound"])
        .reset_index(drop=True)
    )

    print("Writing parquet artifacts...")
    unified_gene_expression.to_parquet(RAW / "train_counts.parquet")
    unified_sample_metadata.to_parquet(RAW / "train_metadata.parquet")
    unified_compound_chemistry.to_parquet(RAW / "train_chemistry.parquet")


def main() -> None:
    if not os.environ.get("TVC_TOKEN"):
        raise SystemExit(
            "TVC_TOKEN not set. Run via: uv run --env-file .env python src/vcpi_ml/download.py"
        )
    RAW.mkdir(parents=True, exist_ok=True)
    download_weights()
    download_counts_metadata_chemistry()
    print(f"Done. Files written to {RAW}")


if __name__ == "__main__":
    main()
