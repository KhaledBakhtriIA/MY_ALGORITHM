# Attention Modifications

- Rewrote `SelfAttention.compute` using NumPy to vectorize similarity score calculation and softmax generation for significant performance boost.
- Added distinct, learnable Query (`W_q`), Key (`W_k`), and Value (`W_v`) projection matrices so gradients flow independently instead of approximating.
- Introduced `SelfAttention.backward` to compute gradient `dX` properly by differentiating outputs down to `Q`, `K`, and `V`. Added step to calculate weight gradients (`dW_q`, `dW_k`, `dW_v`) and apply parameter updates.
- Added internal cache storing `Q`, `K`, `V`, `attention_weights`, etc. for backward optimization.
