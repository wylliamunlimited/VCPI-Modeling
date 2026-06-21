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
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def morgan_matrix(smiles: pd.Series | pd.DataFrame, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
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
    fg_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    encoding = np.stack([fg_generator.GetFingerprintAsNumPy(mol) for mol in mols])
    return encoding
