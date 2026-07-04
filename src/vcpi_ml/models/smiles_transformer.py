import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        assert d_model % n_heads == 0
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Project the full d_model (all heads at once), not just one head's d_k.
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # mixes the concatenated heads

    def forward(self, X: torch.Tensor):
        batch, seq, _ = X.shape

        # Project (all heads at once), then split into heads: (batch, n_heads, seq, d_k).
        # view carves d_model into (n_heads, d_k); transpose moves the head axis in front.
        Q = self.W_Q(X).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(X).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(X).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

        score = (Q @ K.transpose(-2, -1)) / (
            self.d_k**0.5
        )  # (batch, n_heads, seq, seq)
        weights = F.softmax(score, dim=-1)  # (batch, n_heads, seq, seq)
        out = weights @ V  # (batch, n_heads, seq, d_k)

        # Merge heads back to one vector per token, then mix: (batch, seq, d_model).
        out = out.transpose(1, 2).reshape(batch, seq, self.d_model)
        return self.W_O(out)


class SmilesTransformer(nn.Module):
    # TODO (rung 6 phase 2): embed -> +positional -> EncoderBlocks(MultiHeadAttention)
    # -> masked mean-pool -> Linear(d_model, n_genes) regression head.
    pass
