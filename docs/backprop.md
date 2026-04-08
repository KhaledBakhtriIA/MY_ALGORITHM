# Backpropagation Module Documentation

Purpose
- Coordinates the mathematical backward pass.
- Computes the derivative of loss against every variable and propagates it "backwards" through the entire network architecture.
- Calculates and updates the weights at the `output_layer.py` and `embeddings.py` stages iteratively so the network learns (gradient descent).

Available functions
- `calculate_d_logits(logits, target_id)`
  - Computes the mathematical derivative of categorical cross-entropy loss coupled with softmax simply as: *Prediction Probability - 1 (if target)*.
- `backpropagate_step(output_projection, embedding_map, list_of_params, ...)`
  - The step-by-step orchestrator mapping backprop:
    1. Grabs `d_logits` error evaluation.
    2. Runs `d_logits` down into `output_layer.backward` (updates projection mappings).
    3. Traces backwards identically through `synthesizer` and `attention` layers. Currently, these layers don't have trainable parameters (just equations!), so the generated `d_vector` passes down intact to the final layer.
    4. Triggers `embedding_map.update()` to remap the core mathematical vectors representing each word closer to usefulness.

Modification Log (2026-04-08)
1. Read the feature request explicitly detailing the missing embedding update mechanism and requesting the "traces the error back" architecture.
2. Created `backprop.py` containing the error gradient coordinator logic.
3. Updated `embeddings.py` adding the `update()` function to physically mutate the matrix weights.
4. Updated `output_layer.py` adding the `backward()` method to calculate `d_vector` and mathematically perform weight descent against `self.matrix`.
5. Mapped the end-to-end scaffolding required to allow the model to actually "learn" from a token-by-token loss calculation properly.