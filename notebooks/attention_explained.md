# Self-Attention, Explained (my notes)

Companion to `attention.py`. This is the concept walkthrough behind the toy
script — Q/K/V, the dimensions, and the questions I kept tripping over.

---

## 1. The one analogy: a *soft* dictionary lookup

A Python dict does a **hard** lookup: `d[query]` finds the key *exactly* equal
to `query` and returns its one value.

Attention is a **soft, differentiable** lookup:

- the `query` is compared against **every** key by similarity (not exact match),
- those similarities become **weights** (softmax → sum to 1),
- you return a **weighted blend of all the values**.

Each token says: *"here's what I'm looking for (query); let me pull a weighted
mix of everyone's information (values), based on who matches me (keys)."*

---

## 2. Embedding is a SEPARATE step, before attention

There are two different "turn into vectors" steps — don't conflate them:

1. **Input embedding layer** — token ID → first vector, via a lookup table.
   Happens **once, before any attention**. *Not* part of attention.
2. **Q/K/V projections** — `X @ W_Q` etc., re-projecting the already-embedded
   vectors. *This* part **is** inside attention.

```
token IDs → [EMBEDDING LAYER] → +positional → [ ATTENTION (Q/K/V) → FFN ] × N → out
            └ separate, before ┘              └──── attention re-projects & mixes ────┘
```

Attention does **not** create the initial embeddings — it receives and
contextualizes them. This is the same for the **encoder and the decoder**; the
decoder just adds *masking* (can't see future tokens) and *cross-attention*.

---

## 3. Dimensions (the part that confused me most)

`d_model` is **NOT** the number of samples. One input is a **sequence of token
vectors** — a 2D matrix:

| axis | meaning |
|---|---|
| `seq_len` | number of **tokens** in the sequence (rows of X) |
| `d_model` | numbers per token (the **embedding size** / "width") |
| `batch` | number of **samples** (sequences) — a *third* axis, dropped in the toy |
| `d_k` | query/key dimension (inside attention) |
| `d_v` | value dimension (output width); often = `d_k` |

So `X` of shape `(seq_len, d_model)` = `seq_len` tokens stacked as rows, each a
`d_model`-length vector. A full batch is `(batch, seq_len, d_model)`.

Example — SMILES `"CCO"` (ethanol), char-level: tokens `C, C, O` → `seq_len=3`;
embed each into 8 numbers → `d_model=8`; so `X` is `(3, 8)`.

Contrast with the MLP: there one sample was a single vector (2048 fingerprint
bits). In attention, one sample is a `(seq_len × d_model)` **matrix**.

---

## 4. Q, K, V — three roles per token

Each token's embedding is projected three ways:

- **Query (Q):** "what am I looking for?"
- **Key (K):** "how do I advertise myself to be found?"
- **Value (V):** "the actual information I carry."

`Q · K` decides **how much** attention (routing); `V` decides **what content**
flows once matched. K and V are separate so a token can be *matched* on one
thing but *deliver* another.

### Example (coreference)

> "The **animal** didn't cross the **street** because **it** was too tired."

- "it"'s **query** ~ "I'm a pronoun looking for my referent."
- "animal"'s **key** ~ "animate noun"; "street"'s **key** ~ "inanimate location."
- `Q_it · K_animal` is **high** → after softmax, "it" attends mostly to "animal."
- "animal"'s **value** (its meaning) flows into "it"'s output → reference resolved.

### The honest, de-anthropomorphized version

The model never stores English like "I am a noun." It's all floats:

```
"it" → token ID 318 → embedding lookup → [0.23, -1.1, ...]  (512 learned floats)
Q_it = embedding_it @ W_Q                                    (W_Q learned floats)
Q_it · K_animal = 8.3   (just a number)
softmax([8.3, 2.1, ...]) = [0.90, 0.02, ...]
```

`8.3` is not "animacy" — it's what those floats produce. **Gradient descent
tuned `W_Q`/`W_K` until these dot products happened to be useful** (reduced
loss). The "noun looking for referent" is *our reverse-engineered story* about
the emergent geometry, not something stored. The representation is
**distributed** — no single dimension means "animacy." (Attention *weights*,
the seq×seq matrix, ARE often inspectable — coreference heads have been found in
real BERT — but the Q/K/V vectors themselves are opaque.)

---

## 5. `d_k`: Q must match K; V need not

$Q \cdot K$ is a **dot product**, defined only between equal-length vectors, so

$$
\dim(Q) = \dim(K) = d_k \quad\text{is mandatory.}
$$

$V$ is never dotted (only weighted-summed), so it can have its own $d_v$.
Usually $d_k = d_v$ for simplicity, but the only hard rule is
$\dim(Q) = \dim(K)$.

$$
Q \in \mathbb{R}^{\,\text{seq} \times d_k}, \quad
K \in \mathbb{R}^{\,\text{seq} \times d_k}, \quad
V \in \mathbb{R}^{\,\text{seq} \times d_v}
$$

---

## 6. The 4 steps with shapes

For $X \in \mathbb{R}^{\,\text{seq} \times d_{\text{model}}}$, with projection
matrices $W_Q, W_K \in \mathbb{R}^{\,d_{\text{model}} \times d_k}$ and
$W_V \in \mathbb{R}^{\,d_{\text{model}} \times d_v}$:

$$
\textbf{1.}\quad Q = X W_Q,\qquad K = X W_K,\qquad V = X W_V
\qquad (\text{seq} \times d_k,\ \text{seq} \times d_k,\ \text{seq} \times d_v)
$$

$$
\textbf{2.}\quad S = \frac{Q K^{\top}}{\sqrt{d_k}}
\qquad (\text{seq} \times \text{seq},\ d_k \text{ contracts away})
$$

$$
\textbf{3.}\quad A = \operatorname{softmax}(S)\ \text{(row-wise)}
\qquad (\text{each row sums to } 1)
$$

$$
\textbf{4.}\quad \text{output} = A\,V
\qquad (\text{seq} \times d_v,\ \text{seq contracts away})
$$

The whole thing in one line — *scaled dot-product attention*:

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

- $S_{ij}$ = how much token $i$ attends to token $j$.
- output = back to one vector per token, each now enriched with context.

---

## 7. Dot product vs cosine similarity

They're related but not the same:

$$
A \cdot B = \sum_i a_i b_i = \lVert A \rVert\, \lVert B \rVert \cos\theta
\qquad\text{(dot product)}
$$

$$
\cos\theta = \frac{A \cdot B}{\lVert A \rVert\, \lVert B \rVert}
\qquad\text{(cosine similarity = dot product} \div \text{magnitudes)}
$$

Attention uses the **raw dot product**, scaled by a **constant** $1/\sqrt{d_k}$
(*not* by the vectors' magnitudes) — "scaled dot-product attention." So
**magnitude matters**: a token can "shout" with a large-magnitude key. Cosine
would erase that; attention keeps it as usable signal.

### Why $\div \sqrt{d_k}$?

The dot product of two random $d_k$-vectors has standard deviation
$\sim\!\sqrt{d_k}$. Without scaling, large scores push softmax toward
near-one-hot (spiky) $\Rightarrow$ vanishing gradients. Dividing by $\sqrt{d_k}$
keeps the scores in a sane range. (See the "NO scaling" block in `attention.py`
— the weights get visibly spikier.)

Recall softmax itself:

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

---

## 8. How peer tokens change a token's meaning

This is the *whole point* of attention. The output step:

$$
\text{output}_i = \sum_j A_{ij}\, V_j \qquad\text{(sum over ALL tokens } j)
$$

Token *i*'s new vector is a weighted blend of **every** token's value — so it's
reshaped by its peers ("not good" ≠ "good"). The seq×seq weight matrix is the
bookkeeping: row *i* = how token *i* spreads its attention.

**Stacking layers compounds it:** after layer 1 every token already carries some
context; layer 2 attends over those contextualized vectors, so context spreads
further (token *i* indirectly absorbs tokens it didn't directly attend to). A
few layers deep → each token reflects the whole sequence.

---

## 9. Multi-head (the rung-6 detail, not in the toy)

"Multi-head" = run attention $h$ times in parallel, each in a smaller subspace
($d_k = d_{\text{model}} / h$), then concatenate. Each head can learn a
different kind of relationship. "12 heads, 12 layers" = 12 *attention* heads per
layer, 12 layers stacked — heads $\neq$ layers $\neq$ stages.

---

## TL;DR

> Embed tokens into float vectors (separate layer) → project each into Q/K/V →
> score every query against every key with a **scaled dot product** → softmax to
> weights → output each token as a **weighted blend of all values**. Meaning is
> emergent, distributed floats tuned by gradient descent — the English stories
> are ours, layered on top.
