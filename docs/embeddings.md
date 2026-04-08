# Embedding Module Documentation

Purpose
- Converts integer token IDs into vectors consisting of lists of floating-point numbers.
- Gives tokens their mathematical "meaning" and "shape", where mathematically similar concepts will map into vectors that are closer together in space.
- Creates the embedding matrix that maps every token to a vector.

Available classes/objects
- `EmbeddingMap(vocab_size, embedding_dim)`: 
  - Generates the core matrix linking token IDs to spatial dimensions.
- `embed_sequence(token_ids)`:
  - Takes an inputted list of token sequence IDs from the tokenizer and maps them out to floating-point vectors on the embedding matrix.

Future Roadmap (Backpropagation / Learning)
- Currently, embeddings are initialized as random weights and do not update. There is no real meaning encoded yet without training.
- Future implementations will require an `update(self, token_id, gradient, learning_rate)` method to perform gradient descent on the matrix vectors.

Modification Log (2026-04-08)
1. Read the `embeddings.md` instructions and goals.
2. Created standard Python implementations for `embeddings.py` (avoiding complex external dependencies for now). 
3. Created an `EmbeddingMap` class that handles dynamically scaling matrix initialization via Gaussian distribution scaled by dimension count.
4. Set up an `embed_sequence` function mapped to bounds checks (falling back to `<UNK>` token 1 from `tokenizer.py` if needed).
5. Documented modification updates to align with user tracking preferences. 
