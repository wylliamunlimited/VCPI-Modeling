"""smiles_transformer.py — a from-scratch SMILES transformer (rung 6).

Learns chemistry features directly from char-tokenized SMILES (vs. the
hand-designed Morgan fingerprints of rungs 3-4), then predicts the ~13k gene
expression values. Built bottom-up from three small, independently-testable
modules — the attention math is hand-written, not nn.MultiheadAttention:

    SmilesTransformer          embed -> +positional -> encoder blocks -> pool -> head
      └─ EncoderBlock          attention + FFN, each in an Add & Norm residual
            └─ MultiHeadAttention   scaled dot-product attention, n_heads lanes

A padding mask (1=real, 0=pad) is threaded through so padding never influences
attention (pad keys scored -inf before softmax) or the mean-pool (pad tokens
excluded from the average). See notebooks/attention_explained.md for the theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled dot-product self-attention split across n_heads parallel lanes.

    Each head is an independent (Q, K, V) projection into a d_k-wide subspace;
    heads split the *feature width* (d_model = n_heads * d_k), never the tokens —
    every head attends over the whole sequence. The per-head projections are
    packed into one Linear each (W_Q/W_K/W_V of shape d_model x d_model) so it's
    a single matmul; a reshape carves the output back into heads. W_O mixes the
    concatenated heads. See notebooks/attention_explained.md §10 for the why.

    Shape-preserving: (batch, seq, d_model) in -> (batch, seq, d_model) out.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        """d_model = per-token width; n_heads = parallel lanes (must divide d_model).

        Sets d_k = d_model // n_heads (each head's subspace width) and creates the
        four projections. Aim for d_k ~ 32-64: too thin (d_k -> 1) and each head's
        dot product degenerates to a scalar comparison.
        """
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

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        """Contextualize each token by a relevance-weighted blend of all tokens.

        X: (batch, seq, d_model). Returns the same shape, each token now enriched
        with context. Four steps: project + split into heads (batch, n_heads, seq,
        d_k); score = softmax(Q Kᵀ / √d_k) — the (seq, seq) affinity map, "who
        attends to whom"; mix = weights @ V — pull in each token's value content;
        merge heads and project with W_O.

        mask: optional (batch, seq), 1=real / 0=pad. Reshaped to (batch, 1, 1, seq)
        so it broadcasts over heads and query rows; pad *keys* are scored -inf
        before softmax (e^-inf = 0), so no token attends to padding.
        """
        batch, seq, _ = X.shape

        # Project (all heads at once), then split into heads: (batch, n_heads, seq, d_k).
        # view carves d_model into (n_heads, d_k); transpose moves the head axis in front.
        Q = self.W_Q(X).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(X).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(X).view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

        score = (Q @ K.transpose(-2, -1)) / (
            self.d_k**0.5
        )  # (batch, n_heads, seq, seq)
        if mask is not None:
            score = score.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        weights = F.softmax(score, dim=-1)  # (batch, n_heads, seq, seq)
        out = weights @ V  # (batch, n_heads, seq, d_k)

        # Merge heads back to one vector per token, then mix: (batch, seq, d_model).
        out = out.transpose(1, 2).reshape(batch, seq, self.d_model)
        return self.W_O(out)


class EncoderBlock(nn.Module):
    """One transformer encoder layer: attention sublayer + feed-forward sublayer.

    Each sublayer is wrapped in the same Add & Norm pattern (post-norm, matching
    the original Transformer figure): LayerNorm(x + Dropout(Sublayer(x))). The
    attention sublayer *mixes information across tokens*; the feed-forward
    sublayer *transforms each token independently* (the per-token nonlinearity
    attention lacks). Residuals let gradients flow when these blocks are stacked;
    the two LayerNorms are separate (each has its own learned scale/shift), while
    one parameter-free Dropout is shared. Shape-preserving throughout.

    attention: a MultiHeadAttention (token mixing).
    ffn:       a per-token network, e.g. Linear(d_model, h) -> ReLU -> Linear(h, d_model).
    """

    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        dropout: float = 0.1,
        d_model: int = 128,
    ):
        super().__init__()
        self.attn = attention
        self.ffn = ffn
        self.norm_attn = nn.LayerNorm(d_model)  # normalizes the attention sublayer
        self.norm_ffn = nn.LayerNorm(d_model)  # normalizes the feed-forward sublayer
        self.dropout = nn.Dropout(dropout)  # parameter-free, safe to reuse

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """X: (batch, seq, d_model) -> same shape, each token refined.

        Two Add & Norm sublayers: first attention (cross-token mixing), then the
        feed-forward (per-token transform). Dropout is applied to each sublayer's
        output before it rejoins the residual branch.
        """
        # Add & Norm #1 — attention sublayer.
        attn_residual = X + self.dropout(self.attn(X, mask))  # Add (residual)
        attn_hidden = self.norm_attn(attn_residual)  # Norm

        # Add & Norm #2 — feed-forward sublayer.
        ffn_residual = attn_hidden + self.dropout(self.ffn(attn_hidden))  # Add
        return self.norm_ffn(ffn_residual)  # Norm


class SmilesTransformer(nn.Module):
    """End-to-end SMILES -> gene-expression model (rung 6).

    Pipeline: token ids -> embed (+ learned positional) -> n_attention encoder
    blocks -> masked mean-pool over real tokens -> Linear regression head. The
    pooled vector is a *learned molecule embedding* (the transformer analog of a
    Morgan fingerprint); the head maps it to the ~13k gene targets. Same
    fit/predict role as the other models, so a driver can swap it in.
    """

    def __init__(
        self,
        vocab_size: int,
        n_genes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_attention: int = 1,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        """vocab_size = char vocab + 1 (pad id 0); n_genes = regression outputs.

        d_model = token width; n_heads = attention lanes; n_attention = how many
        EncoderBlocks to stack (depth); max_len = longest tokenized SMILES (sizes
        the positional table — keep equal to the tokenizer's max_len).
        """
        super().__init__()
        # Embedding: token id -> vector (pad row zeroed) + a learned per-position vector.
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)  # learned positional encoding
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    attention=MultiHeadAttention(d_model, n_heads),
                    ffn=nn.Sequential(
                        nn.Linear(d_model, 512), nn.ReLU(), nn.Linear(512, d_model)
                    ),
                    dropout=dropout,
                    d_model=d_model,
                )
                for _ in range(n_attention)
            ]
        )  # (batch, seq, d_model)

        self.head = nn.Linear(d_model, n_genes)  ## Regression Head

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        """X: (batch, seq) token ids, mask: (batch, seq) 1=real/0=pad -> (batch, n_genes).

        Embed + add positional, run the encoder blocks (mask threaded to each),
        masked mean-pool to one vector per molecule, then the regression head.
        """
        seq = X.shape[1]  # take seq from (batch, seq)
        assert seq <= self.pos.num_embeddings, (
            f"seq={seq} exceeds max_len={self.pos.num_embeddings}; "
            "truncate tokens or raise max_len"
        )
        positions = torch.arange(seq, device=X.device)
        X = self.embed(X) + self.pos(positions)

        for block in self.blocks:
            X = block(X, mask) # (batch, seq, d_model)

        m = mask.unsqueeze(dim=-1) # (batch, seq, 1)
        X = (X * m).sum(dim=1) / m.sum(dim=1).clamp(min=1) # (batch, d_model)
        X = self.head(X)  # Regression --> (batch, n_genes)
        return X
