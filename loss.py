import math

def cross_entropy_loss(predictions, targets):
    """
    Calculates the cross-entropy loss for a sequence.
    predictions: A list of probability distributions (one list of probabilities per position).
    targets: A list of the correct token IDs for each position.
    
    Returns a single average loss number for the sequence.
    """
    if not predictions or not targets:
        return 0.0
        
    total_loss = 0.0
    steps = max(len(predictions), len(targets))
    epsilon = 1e-15  # Add tiny epsilon to prevent log(0) errors
    
    for i in range(steps):
        pred_idx = min(i, len(predictions) - 1)
        t_idx = min(i, len(targets) - 1)
        
        target_token = targets[t_idx]
        predicted_probability = predictions[pred_idx][target_token]
        
        # for each position: loss = -log(predicted_probability)
        step_loss = -math.log(predicted_probability + epsilon)
        total_loss += step_loss
        
    return total_loss / steps if steps > 0 else 0.0
