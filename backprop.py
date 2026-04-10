from output_layer import softmax
import numpy as np

def calculate_d_logits(logits, target_id):
    """
    Computes the gradient of the cross-entropy loss with respect to the raw logits.
    """
    probs = softmax(logits)
    d_logits = list(probs)
    d_logits[target_id] -= 1.0
    return d_logits

def backpropagate_step_sequence(output_projection, embedding_map, nodes, input_token_ids, vectors, logits_seq, target_ids, learning_rate=0.01):
    """
    Properly handles backpropagation through sequence-level features including attention.
    Since attention processes a sequence, we backpropagate through the whole sequence.
    """
    
    # Step 1: Calculate loss gradient for all logits in the sequence
    d_logits_seq = [calculate_d_logits(l, t) for l, t in zip(logits_seq, target_ids)]
    
    # Step 2: Backpropagate through output projection for each step
    d_vectors = []
    for vec, d_logit in zip(vectors, d_logits_seq):
        d_vec = output_projection.backward(vec, d_logit, learning_rate)
        d_vectors.append(d_vec)
        
    # Step 3: Backpropagate through self-attentions (one per PerspectiveNode)
    # The synthesizer merged the nodes by averaging.
    n_nodes = len(nodes)
    d_vectors_split = [[v / n_nodes for v in vec] for vec in d_vectors]
    
    all_d_attentions = []
    for node in nodes:
        d_attention = node.attention_module.backward(d_vectors_split, learning_rate)
        if d_attention:
            all_d_attentions.append(d_attention)
            
    # Step 4: Update Embeddings
    # Embeddings are shared across nodes, so we sum the gradients
    sum_d_attention = []
    if all_d_attentions:
        seq_len = len(all_d_attentions[0])
        dim = len(all_d_attentions[0][0])
        for i in range(seq_len):
            grad_sum = [0.0] * dim
            for d_att in all_d_attentions:
                if i < len(d_att):
                    for d in range(dim):
                        grad_sum[d] += d_att[i][d]
            sum_d_attention.append(grad_sum)
            
    for t_id, d_emb in zip(input_token_ids, sum_d_attention):
        embedding_map.update(t_id, d_emb, learning_rate)

    return sum_d_attention

