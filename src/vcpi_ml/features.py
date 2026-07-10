"""features.py — turn molecules (SMILES) into model inputs X.

The chemistry side of the pipeline: where a compound's structure becomes a
numeric feature vector a model can learn from. Rung 2 ignored chemistry
entirely; this is the first module that reads it.

Morgan / ECFP fingerprints encode each molecule as a fixed-length bit vector,
one bit per circular substructure ("does this molecule contain this atom +
its neighborhood out to `radius`?"). That bit vector is the X a linear model
(Ridge) maps to per-gene expression.
"""

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModel
from vcpi_ml.device import DEVICE


def morgan_matrix(
    smiles: pd.Series | pd.DataFrame, n_bits: int = 2048, radius: int = 2
) -> np.ndarray:
    """Encode SMILES as a Morgan-fingerprint feature matrix.

    Parameters
    ----------
    smiles
        Iterable of SMILES strings, one per compound. Row order is preserved,
        so the caller must keep it aligned with the expression matrix's
        compound order.
    n_bits
        Fingerprint length (feature dimension); each bit = one substructure.
    radius
        How far around each atom the substructures reach (ECFP radius).

    Returns
    -------
    np.ndarray of shape (n_compounds x n_bits), 0/1 substructure presence.

    Note: assumes every SMILES parses (MolFromSmiles is not None); add None
    handling here if that assumption breaks.
    """
    fg_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    encoding = np.stack([fg_generator.GetFingerprintAsNumPy(mol) for mol in mols])
    return encoding


def chemberta_matrix(
    smiles: pd.Series | pd.DataFrame,
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """Encode SMILES as frozen ChemBERTa embeddings (a learned featurizer).

    The pretrained-transformer analog of morgan_matrix: instead of hand-designed
    substructure bits, each molecule becomes a 768-dim vector from ChemBERTa —
    a RoBERTa transformer pretrained on ~millions of SMILES. The model is
    *frozen* (eval + no_grad): it's a fixed SMILES -> vector function, so nothing
    trains here. Run once and cache; the output never changes.

    Uses ChemBERTa's own tokenizer (its ids must match its pretraining) and a
    masked mean-pool over real tokens (the same pooling as the from-scratch
    transformer) to collapse the per-token states into one vector per molecule.

    Parameters
    ----------
    smiles
        Iterable of SMILES strings, one per compound. Row order is preserved,
        so keep it aligned with the expression matrix's compound order.
    model_name
        HuggingFace model id (default: the classic ZINC-pretrained ChemBERTa).
    batch_size
        Compounds per forward pass (all ~14k won't fit at once).
    max_length
        Truncate tokenized SMILES to this many tokens (plenty for these).

    Returns
    -------
    np.ndarray of shape (n_compounds x 768), the pooled embeddings — feed
    straight to Ridge/MLP like a fingerprint (it already IS a vector).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    embeddings = []
    for i in range(0, len(smiles), batch_size):
        batch = list(smiles[i : i + batch_size])  # list of <=batch_size SMILES
        # Tokenize with ChemBERTa's tokenizer; padding=True pads to the batch's
        # longest, so seq <= max_length (varies per batch).
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(DEVICE)  # (batch, seq)
        with torch.no_grad(): # frozen
            out = model(**enc).last_hidden_state  # (batch, seq, 768)
        m = enc.attention_mask.unsqueeze(-1)       # (batch, seq, 1)
        # Masked mean-pool over real tokens -> one vector per molecule.
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)  # (batch, 768)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings).numpy()  # (n_compounds, 768)