import numpy as np

class SelfAttention:
    """
    Core intelligence module that computes relevance (resemblance) 
    between tokens across the data sequence using scaled dot-product attention
    optimized with NumPy.
    """
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        # Initialize learnable weights for Q, K, V
        limit = np.sqrt(6 / (embedding_dim + embedding_dim))
        self.W_q = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))
        self.W_k = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))
        self.W_v = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))
        self.cache = {} # Cache for backward pass
        
    def compute(self, vectors):
        """
        Given a sequence of vectors, computes the self-attention representation.
        Returns a new sequence of vectors of the same dimension.
        """
        if len(vectors) == 0:
            return []
            
        x = np.array(vectors) # (seq_len, d_k)
        
        # Linear projections
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        scale = np.sqrt(Q.shape[-1])
        
        # 1. Similarity Scores
        scores = np.dot(Q, K.T) / scale # (seq_len, seq_len)
        
        # 2. Softmax
        # numerically stable softmax along the last axis
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 3. Context Vectors
        context = np.dot(attention_weights, V) # (seq_len, d_k)
        
        output_pre_norm = context + x  # residual connection
        
        # layer normalization
        mean = np.mean(output_pre_norm, axis=-1, keepdims=True)
        var = np.var(output_pre_norm, axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-8)
        output = (output_pre_norm - mean) / std
        
        # Cache for backprop
        self.cache = {
            "x": x, "Q": Q, "K": K, "V": V,
            "attention_weights": attention_weights,
            "scale": scale,
            "output_pre_norm": output_pre_norm,
            "mean": mean,
            "std": std
        }
        
        return output.tolist()

    def backward(self, d_out, learning_rate):
        """
        Proper backpropagation through the self-attention layer.
        """
        d_out = np.array(d_out)
        if d_out.size == 0 or not self.cache:
            return []
            
        x = self.cache["x"]
        
        # Ensure d_out matches the sequence length of x
        if d_out.shape[0] < x.shape[0]:
            padding = np.zeros((x.shape[0] - d_out.shape[0], d_out.shape[1]))
            d_out = np.vstack([d_out, padding])
        elif d_out.shape[0] > x.shape[0]:
            d_out = d_out[:x.shape[0], :]
        Q = self.cache["Q"]
        K = self.cache["K"]
        V = self.cache["V"]
        P = self.cache["attention_weights"]
        scale = self.cache["scale"]
        output_pre_norm = self.cache["output_pre_norm"]
        mean = self.cache["mean"]
        std = self.cache["std"]
        
        N = x.shape[-1]
        
        # d_out is dL/dY, shape (seq_len, d_k)
        # Backprop through layer normalization
        # Let's say dY = d_out
        # dX_norm component
        d_out_pre_norm = (1. / std) * (
            d_out - np.mean(d_out, axis=-1, keepdims=True) - 
            ((output_pre_norm - mean) / (std ** 2)) * np.mean(d_out * (output_pre_norm - mean), axis=-1, keepdims=True)
        )
        
        # Backprop through residual connection
        # dL/dcontext = d_out_pre_norm
        # dL/dx = d_out_pre_norm (plus updates from Q, K, V later)
        d_context = d_out_pre_norm
        
        # dL/dV
        dV = np.dot(P.T, d_context)
        
        # dL/dP 
        dP = np.dot(d_context, V.T)
        
        # dL/dS (where S is the pre-softmax scores)
        sum_dP_P = np.sum(dP * P, axis=-1, keepdims=True)
        dS = P * (dP - sum_dP_P)
        
        # dL/dQ and dL/dK
        dQ = np.dot(dS, K) / scale
        dK = np.dot(dS.T, Q) / scale
        
        # Weight gradients
        dW_q = np.dot(x.T, dQ)
        dW_k = np.dot(x.T, dK)
        dW_v = np.dot(x.T, dV)
        
        # Gradient clipping (analogous to torch.nn.utils.clip_grad_norm_)
        max_norm = 1.0
        # Compute global norm of these gradients
        total_norm = np.sqrt(np.sum(dW_q**2) + np.sum(dW_k**2) + np.sum(dW_v**2) + 1e-6)
        # Uncomment to debug vanishing gradients: 
        # print(f"Attention Total Grad Norm: {total_norm:.6f}")
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            dW_q *= clip_coef
            dW_k *= clip_coef
            dW_v *= clip_coef
        
        # Input gradients (Residual path + Q,K,V path)
        dX = d_out_pre_norm + np.dot(dQ, self.W_q.T) + np.dot(dK, self.W_k.T) + np.dot(dV, self.W_v.T)
        
        # Update weights
        self.W_q -= learning_rate * dW_q
        self.W_k -= learning_rate * dW_k
        self.W_v -= learning_rate * dW_v
        
        return dX.tolist()

