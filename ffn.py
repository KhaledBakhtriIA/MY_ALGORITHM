import numpy as np

class FeedForward:
    """
    A simple Feed-Forward Network with ReLU activation.
    Provides non-linearity to the model, allowing it to learn complex mappings
    beyond simple linear transformations.
    """
    def __init__(self, embedding_dim, hidden_dim=None):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embedding_dim * 4
        
        # Initialize weights with Xavier/Glorot scaling
        limit1 = np.sqrt(6 / (embedding_dim + self.hidden_dim))
        self.W1 = np.random.uniform(-limit1, limit1, (embedding_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        
        limit2 = np.sqrt(6 / (self.hidden_dim + embedding_dim))
        self.W2 = np.random.uniform(-limit2, limit2, (self.hidden_dim, embedding_dim))
        self.b2 = np.zeros(embedding_dim)
        
        self.cache = {}
        
    def compute(self, x):
        """
        Forward pass: x -> Linear 1 -> ReLU -> Linear 2 -> Add & Norm -> output
        """
        x_np = np.array(x)
        if x_np.size == 0:
            return []
            
        # Linear 1
        z1 = np.dot(x_np, self.W1) + self.b1
        
        # ReLU activation
        a1 = np.maximum(0, z1)
        
        # Linear 2
        z2 = np.dot(a1, self.W2) + self.b2
        
        # Residual connection
        output = z2 + x_np
        
        self.cache = {
            "x": x_np,
            "z1": z1,
            "a1": a1,
            "output": output
        }
        
        return output.tolist()

    def backward(self, d_out, learning_rate):
        """
        Backward pass through the Feed-Forward layer.
        """
        d_out = np.array(d_out)
        if d_out.size == 0 or not self.cache:
            return []
            
        x = self.cache["x"]
        a1 = self.cache["a1"]
        z1 = self.cache["z1"]
        
        # Ensure d_out matches seq_len of x
        if d_out.shape[0] < x.shape[0]:
            padding = np.zeros((x.shape[0] - d_out.shape[0], d_out.shape[1]))
            d_out = np.vstack([d_out, padding])
        elif d_out.shape[0] > x.shape[0]:
            d_out = d_out[:x.shape[0], :]
            
        # The output is z2 + x, so gradient branches
        d_z2 = d_out  # (seq_len, embedding_dim)
        
        # Gradient for W2 and b2
        d_W2 = np.dot(a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)
        
        # Gradient passing through W2 to the ReLU layer
        d_a1 = np.dot(d_z2, self.W2.T)
        
        # Gradient through ReLU (derivative of ReLU is 1 for z1 > 0, else 0)
        d_z1 = d_a1.copy()
        d_z1[z1 <= 0] = 0
        
        # Gradient for W1 and b1
        d_W1 = np.dot(x.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)
        
        # Gradient passing through W1 to the inputs
        d_x = np.dot(d_z1, self.W1.T)
        
        # Add residual gradient (from the + x connection)
        d_x_total = d_x + d_out
        
        # Gradient Clipping (L2 Norm)
        max_norm = 1.0
        for grad in [d_W1, d_b1, d_W2, d_b2]:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad *= (max_norm / norm)
        
        # Update weights (Gradient Descent)
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        
        return d_x_total.tolist()