
import pandas as pd
import numpy as np
import torch as torch
import torch.nn as nn
from tqdm import tqdm


class MLP(nn.Module):

    def __init__(self, n_in: int = 2048, n_out: int = 12995, hidden: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPModel():

    def __init__(
        self, 
        n_in: int = 2048, n_out: int = 12995, hidden: int = 512,
        lr: float = 1e-3
        ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = MLP(n_in=n_in, hidden=hidden, n_out=n_out)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()


    def fit(self, X: np.ndarray, Y: pd.DataFrame, epoch: int = 500, batch: int | None = None):

        X, Y = (
            torch.tensor(X, dtype=torch.float32, device=self.device), 
            torch.tensor(Y.to_numpy(), dtype=torch.float32, device=self.device)
        )
        n = X.shape[0]

        for ep in tqdm(range(epoch)):
            perm = torch.randperm(n, device=self.device)
            if batch:
                for start in range(0, n, batch):
                    idx = perm[start: start + batch]
                    self.optimizer.zero_grad()
                    xb, yb = X[idx], Y[idx]
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)
                    loss.backward()
                    self.optimizer.step()
            else:
                self.optimizer.zero_grad()
                pred = self.model(X)
                loss = self.loss_fn(pred, Y)
                loss.backward()
                self.optimizer.step()

        return self


    def predict(self, X: np.ndarray):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(X)
        return pred.detach().cpu().numpy()