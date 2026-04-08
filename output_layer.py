import math
import random
from tokenizer import decode, build_id_to_token

def softmax(logits):
    if not logits:
        return []
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits)
    return [l / sum_exp for l in exp_logits]

def sample_token(probabilities, temperature=1.0):
    # scale probabilities by temperature
    scaled = [p ** (1.0 / temperature) for p in probabilities]
    total = sum(scaled)
    normalized = [p / total for p in scaled]
    
    # random sample instead of always picking max
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(normalized):
        cumulative += p
        if r <= cumulative:
            return i
    return len(normalized) - 1

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

    def backward(self, vector, d_logits, learning_rate):
        """
        Backpropagates through the projection layer.
        Given the gradient of the loss with respect to logits (d_logits),
        it updates the projection matrix and calculates the gradient with 
        respect to the input vector (d_vector) to pass backward.
        """
        d_vector = [0.0] * self.embedding_dim
        
        # We need to evaluate every vocabulary logit derivative
        for v_idx in range(self.vocab_size):
            d_l = d_logits[v_idx]
            
            for dim in range(self.embedding_dim):
                # Accumulate the gradient for the incoming vector
                d_vector[dim] += d_l * self.matrix[dim][v_idx]
                
                # Perform the gradient descent update on the weights!
                # Weight gradient = d_logit * input_vector
                self.matrix[dim][v_idx] -= learning_rate * d_l * vector[dim]
                
        return d_vector

    def generate_tokens(self, vectors, vocab, temperature=1.0):
        """
        Takes the synthesized output vectors and selects tokens dynamically
        using temperature sampling to construct the final output phrase.
        """
        id_to_token = build_id_to_token(vocab)
        output_ids = []
        
        for vector in vectors:
            logits = self.project(vector)
            probabilities = softmax(logits)
            # Use temperature sampling to avoid greedy collapse (e.g. constant repeats)
            best_id = sample_token(probabilities, temperature=temperature)
            output_ids.append(best_id)
            
        return decode(output_ids, id_to_token)