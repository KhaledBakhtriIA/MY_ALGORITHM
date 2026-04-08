# Attention Module Documentation

Purpose
- Acts as the core intelligence of the network.
- Given a sequence of vectors, it figures out which parts of the data are relevant to other parts mathematically.
- Calculates the "bonds" or "resemblance" between tokens across the provided sequence data.

Available classes/objects
- `SelfAttention(embedding_dim)`: 
  - Central class managing the attention mechanism setup.
- `compute(vectors)`:
  - Takes an inputted sequence of floating-point vectors and applies scaled dot-product self-attention, returning a contextualized output sequence of the same dimension.
  - Implements standard Query-Key-Value mathematical similarities using dot products and softmax.

Modification Log (2026-04-08)
1. Read the initial `attention.md` notes describing the goal of mathematical bonds and resemblance.
2. Implemented `attention.py` avoiding complex dependencies (pure Python with `math` module).
3. Created helper functions `dot_product` and `softmax` (with max-score subtraction for numerical stability).
4. Created the `SelfAttention` class with a `compute()` method to execute a parameter-less scaled dot-product attention mapping.
5. Updated `attention.md` documentation to comprehensively track the module's responsibilities and align with user logging preferences. 