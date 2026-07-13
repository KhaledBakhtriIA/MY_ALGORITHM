import math
import random
import numpy as np

class PositionalEncoding:
    """Implement PyTorch-like Positional Encoding statically via Numpy"""
    def __init__(self, embedding_dim, max_seq_len=50):
        pe = np.zeros((max_seq_len, embedding_dim))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe

class EmbeddingMap:
    """
    Creates and manages the embedding matrix that maps every token ID to a continuous vector in space.
    """
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.matrix = self._initialize_matrix(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
    def _initialize_matrix(self, vocab_size, embedding_dim):
        """
        Initializes the matrix mapping token IDs to floating-point vectors using a Gaussian distribution.
        """
        matrix = []
        for _ in range(vocab_size):
            # Scale the variance based on embedding dimensions 
            std_dev = 1.0 / math.sqrt(embedding_dim) if embedding_dim > 0 else 1.0
            vector = [random.gauss(0, std_dev) for _ in range(embedding_dim)]
            matrix.append(vector)
        return matrix
        
    def embed_sequence(self, token_ids):
        """
        Converts a sequence (list) of token IDs into a sequence of floating-point vectors.
        """
        vectors = []
        for i, token_id in enumerate(token_ids):
            # Fall back to the <UNK> token ID 1 if the token_id is out of bounds
            if token_id < 0 or token_id >= self.vocab_size:
                token_id = 1
            
            # Fetch raw embedding
            raw_vec = self.matrix[token_id]
            
            # Sequence PE fingerprint integration
            if i < len(self.positional_encoding.pe):
                pe_vec = self.positional_encoding.pe[i]
                # Scale word embeddings by sqrt(dim) before adding PE
                scale = math.sqrt(self.embedding_dim)
                final_vec = [raw_vec[d] * scale + float(pe_vec[d]) for d in range(self.embedding_dim)]
            else:
                scale = math.sqrt(self.embedding_dim)
                final_vec = [v * scale for v in raw_vec]
                
            vectors.append(final_vec)
            
        return vectors

    def update(self, token_id, gradient, learning_rate):
        """
        Backpropagation update step.
        Adjusts the embedding vector for a specific token based on the error gradient.
        """
        # Ensure we stay in bounds just like the forward pass
        if token_id < 0 or token_id >= self.vocab_size:
            token_id = 1
            
        scale = math.sqrt(self.embedding_dim)
        for dim in range(self.embedding_dim):
            self.matrix[token_id][dim] -= learning_rate * gradient[dim] * scale
