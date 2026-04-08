from output_layer import softmax

def calculate_d_logits(logits, target_id):
    """
    Computes the gradient of the cross-entropy loss with respect to the raw logits.
    Mathematically, for softmax with cross-entropy, the gradient simplifies to:
    dL/dLogits = (Softmax Probabilities) - (1 if Correct Target Else 0)
    """
    probs = softmax(logits)
    d_logits = list(probs)
    # The target class gets -1
    d_logits[target_id] -= 1.0
    return d_logits

def backpropagate_step(output_projection, embedding_map, input_token_id, vector, logits, target_id, learning_rate=0.01):
    """
    Traces the error backwards step-by-step through the network components.
    This orchestrates the gradient descent calculation and weight updates.

    Path taken back:
    1. Loss -> calculates error against Target ID (`calculate_d_logits`)
    2. Output Layer -> takes d_logits, updates matrix, and returns d_vector
    3. Synthesizer & Node/Attention -> (For now, simplified pass-through since attention lacks internal weights yet)
    4. Embeddings -> uses the unrolled d_vector to update original token weights
    """
    
    # Step 1: Calculate the error at the absolute end (Loss vs Prediction)
    d_logits = calculate_d_logits(logits, target_id)
    
    # Step 2: Backpropagate through the output classification head.
    # This automatically updates the `OutputProjection.matrix` and computes
    # the gradient of the error with respect to the input representation vector.
    d_vector = output_projection.backward(vector, d_logits, learning_rate)
    
    # Step 3: Gradient passes backwards through Synthesizer and Attention layers.
    # Since our SelfAttention implementation presently only calculates dot product similarities 
    # without trainable query/key/value projection matrices...
    # The gradient passes straight through back into the initial embeddings stage map.
    d_embedding = d_vector
    
    # Step 4: Update the original token embeddings so the token representation gets "smarter".
    embedding_map.update(input_token_id, d_embedding, learning_rate)

    return d_vector
