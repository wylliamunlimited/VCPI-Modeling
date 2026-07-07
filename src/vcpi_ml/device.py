"""device.py — shared torch device selection, resolved once at import.

Every model (MLP, transformer) trains on the same device; picking it lives here
so the choice isn't duplicated per model. Import DEVICE and move tensors/modules
onto it with .to(DEVICE).
"""

import torch

# cuda (other machines) → mps (this Mac's GPU) → cpu; resolved once at import.
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
