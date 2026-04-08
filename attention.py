import math

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def softmax(scores):
    if not scores:
        return []
    max_score = max(scores)  # for numerical stability
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    return [s / sum_exp for s in exp_scores]

class SelfAttention:
    """
    Core intelligence module that computes relevance (resemblance) 
    between tokens across the data sequence using scaled dot-product attention.
    """
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        
    def compute(self, vectors):
        """
        Given a sequence of vectors, computes the self-attention representation.
        Returns a new sequence of vectors of the same dimension.
        """
        seq_len = len(vectors)
        if seq_len == 0:
            return []
            
        d_k = len(vectors[0])
        scale = math.sqrt(d_k) if d_k > 0 else 1.0
        
        output_vectors = []
        
        # For each token vector (acting as Query)
        for i in range(seq_len):
            query = vectors[i]
            
            # 1. Compute resemblance scores with all other tokens (Keys)
            raw_scores = []
            for j in range(seq_len):
                key = vectors[j]
                score = dot_product(query, key) / scale
                raw_scores.append(score)
                
            # 2. Apply softmax to get normalized attention weights
            attention_weights = softmax(raw_scores)
            
            # 3. Compute weighted sum of all tokens (Values)
            context_vector = [0.0] * d_k
            for j in range(seq_len):
                weight = attention_weights[j]
                value = vectors[j]
                for dim in range(d_k):
                    context_vector[dim] += weight * value[dim]
                    
            output_vectors.append(context_vector)
            
        return output_vectors