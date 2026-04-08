import math

def cross_entropy_loss(logits, target_id):
    """
    Calculates the cross-entropy loss for a single token prediction.
    Measures how 'wrong' the raw logits are compared to the actual target token ID.
    Higher values mean the model was more wrong. Values close to 0 mean it was very confident and correct.
    """
    if not logits:
        return 0.0
        
    # 1. Calculate softmax probabilities (with max subtraction for numerical stability)
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits)
    
    # Probability assigned to the correct target class
    p_target = exp_logits[target_id] / sum_exp
    
    # 2. Add tiny epsilon to prevent log(0) errors
    epsilon = 1e-15
    
    # 3. Return negative log likelihood
    return -math.log(p_target + epsilon)


def sequence_loss(sequence_logits, target_ids):
    """
    Calculates the average cross-entropy loss across an entire sequence.
    """
    if not sequence_logits or not target_ids:
        return 0.0
    
    total_loss = 0.0
    valid_steps = min(len(sequence_logits), len(target_ids))
    
    for i in range(valid_steps):
        step_loss = cross_entropy_loss(sequence_logits[i], target_ids[i])
        total_loss += step_loss
        
    return total_loss / valid_steps if valid_steps > 0 else 0.0
