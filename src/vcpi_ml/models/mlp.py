"""mlp.py — a PyTorch MLP over Morgan fingerprints (rung 4, first neural net).

Same fingerprint X → expression Y task as ridge.py, but with a 2-hidden-layer
ReLU network and a hand-written training loop (forward → loss → backward →
step) instead of a closed-form solver. Tuned, it ties Ridge (~0.568) — the
fingerprint→expression signal is largely linear, so extra capacity adds little.

Two classes, mirroring the rest of the project:
  - MLP       — the nn.Module (architecture only).
  - MLPModel  — the fit/predict wrapper holding the optimizer + training loop,
                so the driver can swap it for any other model in one line.

Following "DataFrames at the boundary, arrays in the model": fit/predict take
numpy X (Y as a DataFrame for fit), predict returns a raw (n x genes) array
that the driver labels before scoring.
"""

import pandas as pd
import numpy as np
import torch as torch
import torch.nn as nn
from tqdm import tqdm

from vcpi_ml.device import DEVICE


class MLP(nn.Module):
    """The network: n_in → 512 → 512 → n_out, ReLU between layers.

    The ReLUs are what make this more than Ridge — without a nonlinearity the
    stacked Linears collapse to a single linear map (i.e. Ridge). No activation
    on the output (regression).
    """

    def __init__(self, n_in: int = 2048, n_out: int = 12995, hidden: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, n_in) → (batch, n_out). No no_grad here —
        autograd needs to record this graph so loss.backward() can work."""
        return self.model(x)


class MLPModel:
    """fit/predict wrapper around MLP: owns the optimizer, loss, and train loop.

    Hyperparameters (set here, tuned via the sweep): lr and weight_decay (Adam's
    L2 — the MLP's analog of Ridge's alpha; on this data it only hurt, so default
    0). The architecture knobs (hidden) pass through to MLP.
    """

    def __init__(
        self,
        n_in: int = 2048,
        n_out: int = 12995,
        hidden: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        self.device = DEVICE
        self.model = MLP(n_in=n_in, hidden=hidden, n_out=n_out)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()

    def fit(
        self,
        X: np.ndarray,
        Y: pd.DataFrame,
        epoch: int = 500,
        batch: int = 256,
        patience: int = 50,
        min_delta: float = 1e-4,
    ):
        """Train with mini-batch gradient descent; early-stop on a loss plateau.

        Each epoch shuffles, then walks the data in `batch`-sized chunks running
        the 5-line loop (zero_grad → forward → loss → backward → step). Records
        mean epoch loss in self.history. Stops early if the loss hasn't improved
        by `min_delta` for `patience` epochs. Returns self (chainable).

        X: (n, n_in) numpy fingerprints; Y: (n, n_out) expression DataFrame.
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(Y):
            Y = torch.tensor(Y.to_numpy(), dtype=torch.float32, device=self.device)
        n = X.shape[0]
        self.history = []

        best_loss, no_improve = float("inf"), 0
        pbar = tqdm(range(epoch))
        for ep in pbar:
            perm = torch.randperm(n, device=self.device)
            epoch_loss, n_batch = 0.0, 0
            for start in range(0, n, batch):
                idx = perm[start : start + batch]
                self.optimizer.zero_grad()
                xb, yb = X[idx], Y[idx]
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                epoch_loss += loss.item()
                n_batch += 1
                loss.backward()
                self.optimizer.step()
            self.history.append(epoch_loss / n_batch)
            pbar.set_postfix(loss=f"{self.history[-1]:.4f}")

            if self.history[-1] < best_loss - min_delta:
                best_loss, no_improve = self.history[-1], 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    pbar.set_postfix(loss=f"{self.history[-1]:.4f}", stopped=f"ep{ep}")
                    break

        return self

    def predict(self, X: np.ndarray):
        """Inference: (n, n_in) numpy → (n, n_out) numpy.

        eval() + no_grad() turn off training behavior and gradient tracking;
        the result is moved back to cpu/numpy for the driver to label & score.
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(X)
        return pred.detach().cpu().numpy()
