"""attention.py — self-attention from scratch on a toy tensor (rung 5).

A small, runnable demonstration of scaled dot-product self-attention with NO
library magic and NO bio data — purely to make Q/K/V + softmax concrete. See
the companion `attention_explained.md` for the concepts behind every line.

The 4 steps, for ONE sequence X of shape (seq_len, d_model):
    1. project:  Q, K, V = X @ W_Q, X @ W_K, X @ W_V    # (seq_len, d_k) each
    2. score:    scores  = Q @ K.T / sqrt(d_k)          # (seq_len, seq_len)
    3. weight:   weights = softmax(scores, dim=-1)      # each row sums to 1
    4. mix:      output  = weights @ V                  # (seq_len, d_k)

Run: uv run python notebooks/attention.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# Dimensions (this is ONE sequence, not a batch of samples):
#   seq_len = number of tokens          (rows of X)
#   d_model = numbers per token         (the embedding size — NOT #samples)
#   d_k     = query/key dimension       (Q and K must share this; V could differ)
seq_len, d_model, d_k = 10, 8, 8

# Toy "sequence": 10 token embeddings. Make token 5 a copy of token 0 so we can
# verify that identical tokens produce (near-)identical attention rows.
X = torch.randn(seq_len, d_model)
X[5] = X[0]

# Projection matrices: LEARNED in a real transformer, random (untrained) here.
# So the math below is identical to a real model's — only these weights differ.
W_Q = torch.randn(d_model, d_k)
W_K = torch.randn(d_model, d_k)
W_V = torch.randn(d_model, d_k)

# Step 1: project each token into query / key / value vectors.
Q, K, V = X @ W_Q, X @ W_K, X @ W_V          # each (seq_len, d_k)

# Step 2: every query dotted with every key, scaled by sqrt(d_k).
# Use Q (the query projection), not X — attention compares Q to K, not raw input.
scores = Q @ K.T / (d_k ** 0.5)              # (seq_len, seq_len)

# Step 3: softmax each row -> attention weights (each row is a distribution).
weights = F.softmax(scores, dim=-1)

print("= = = = = Weights (scaled by sqrt(d_k)) = = = = =")
print(weights)
print("==> Shape:", weights.shape)           # (seq_len, seq_len)
print("==> Row sums (should all be ~1.0):", weights.sum(dim=-1))
print("==> Rows 0 and 5 are identical tokens -> rows should match:")
print(weights[0])
print(weights[5])
print()

# Step 4: each token's output = weighted blend of ALL tokens' values.
# This is how a token's representation absorbs context from its peers.
outputs = weights @ V
print("= = = = = Outputs = = = = =")
print(outputs)
print("==> Shape:", outputs.shape)           # (seq_len, d_k)
print()

# - - - Why the sqrt(d_k) scaling? Compare against NO scaling - - -
# Without dividing, larger raw scores push softmax toward near-one-hot (spiky),
# which shrinks gradients. The scaling keeps the distribution sane.
print("- - - - - NO scaling (raw Q @ K.T): weights get spikier - - - - -")
scores_raw = Q @ K.T                          # no division
weights_raw = F.softmax(scores_raw, dim=-1)
print(weights_raw)
print("==> compare the sharpness to the scaled weights above")
