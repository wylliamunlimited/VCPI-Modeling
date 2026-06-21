
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def morgan_matrix(smiles: pd.Series | pd.DataFrame, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Turning smiles into Morgan Fingerprint encoding. Output: boolean matrix (compound_count x n_bits)"""
    fg_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    encoding = np.stack([fg_generator.GetFingerprintAsNumPy(mol) for mol in mols])
    return encoding
