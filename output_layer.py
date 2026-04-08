import math
import random
from tokenizer import decode, build_id_to_token

class OutputProjection:
    """
    Final step of the forward pass: maps the generated continuous vectors 
    back into the vocabulary space to produce actual output tokens.
    """
    def __init__(self, embedding_dim, vocab_size):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.matrix = self._initialize_matrix(embedding_dim, vocab_size)

    def _initialize_matrix(self, embedding_dim, vocab_size):
        """
        Initializes the linear projection weights mapping dimensions to vocab logits.
        """
        matrix = []
        for _ in range(embedding_dim):
            std_dev = 1.0 / math.sqrt(vocab_size) if vocab_size > 0 else 1.0
            row = [random.gauss(0, std_dev) for _ in range(vocab_size)]
            matrix.append(row)
        return matrix

    def project(self, vector):
        """
        Pass a single embedding vector through the output layer to compute logits
        for every word in the vocabulary.
        """
        logits = [0.0] * self.vocab_size
        for dim in range(self.embedding_dim):
            weight_row = self.matrix[dim]
            val = vector[dim]
            for v_idx in range(self.vocab_size):
                logits[v_idx] += val * weight_row[v_idx]
        return logits

    def generate_tokens(self, vectors, vocab):
        """
        Takes the synthesized output vectors and selects the most likely 
        token per position (argmax) to construct the final output phrase.
        """
        id_to_token = build_id_to_token(vocab)
        output_ids = []
        
        for vector in vectors:
            logits = self.project(vector)
            # Find the ID with the highest matching logit (argmax)
            best_id = max(range(len(logits)), key=lambda i: logits[i])
            output_ids.append(best_id)
            
        return decode(output_ids, id_to_token)