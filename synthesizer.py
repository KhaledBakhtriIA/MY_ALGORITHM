def synthesize(node_results):
    """
    Takes the output dictionaries from multiple nodes and merges them together.
    For the moment, this performs an averaging operation across the vector sequences.
    Future versions will be expanded to detect tension, contradiction, and agreements.
    """
    if not node_results:
        return []

    # Determine the maximum sequence length across all node outputs to handle 
    # variations caused by different system prompt lengths.
    max_seq_len = max((len(res.get("output_vectors", [])) for res in node_results), default=0)
    
    # Determine the vector embedding dimension
    d_k = 0
    for res in node_results:
        vecs = res.get("output_vectors", [])
        if vecs and len(vecs[0]) > 0:
            d_k = len(vecs[0])
            break
            
    if d_k == 0 or max_seq_len == 0:
        return []

    synthesized_vectors = []
    
    # Position-wise averaging across all node perspectives
    for i in range(max_seq_len):
        avg_vector = [0.0] * d_k
        active_nodes_at_i = 0
        
        for res in node_results:
            vectors = res.get("output_vectors", [])
            if i < len(vectors):
                for dim in range(d_k):
                    avg_vector[dim] += vectors[i][dim]
                active_nodes_at_i += 1
                
        # Calculate the mean for this token position across the perspectives that had a token here
        if active_nodes_at_i > 0:
            avg_vector = [val / active_nodes_at_i for val in avg_vector]
            
        synthesized_vectors.append(avg_vector)

    return synthesized_vectors