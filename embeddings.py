import math
import random


class EmbeddingMap:
    """
    Creates and manages the embedding matrix that maps every token ID to a continuous vector in space.
    """
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.matrix = self._initialize_matrix(vocab_size, embedding_dim)
        
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
        for token_id in token_ids:
            # Fall back to the <UNK> token ID 1 if the token_id is out of bounds
            if token_id < 0 or token_id >= self.vocab_size:
                token_id = 1
                
            vectors.append(self.matrix[token_id])
            
        return vectors
