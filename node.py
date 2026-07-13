import numpy as np
from tokenizer import encode
from embeddings import EmbeddingMap
from attention import SelfAttention
from ffn import FeedForward

class PerspectiveNode:
    """
    A single perspective engine that takes a system prompt and data, 
    processes them through the embedding, attention, and feed-forward modules, 
    and returns a unique reading of the data.
    """
    def __init__(self, name, system_prompt, vocab, embedding_map, attention_module, feed_forward_module=None):
        self.name = name
        self.system_prompt = system_prompt
        self.vocab = vocab
        self.embedding_map = embedding_map
        self.attention_module = attention_module
        
        # Give each node its own specific FFN, or a shared one (if provided)
        self.feed_forward_module = feed_forward_module if feed_forward_module is not None else FeedForward(embedding_map.embedding_dim)

    def process(self, data):
        """
        Runs the combined system prompt and input data through 
        tokenization, embedding, attention, and FFN to generate a perspective output.
        """
        # Combine system prompt with input data
        combined_text = f"{self.system_prompt} {data}"
        
        # 1. Tokenize (convert to integers)
        token_ids = encode(combined_text, self.vocab)
        
        # 2. Embed (convert integers to vectors)
        vectors = self.embedding_map.embed_sequence(token_ids)
        
        # 3. Apply Attention (calculate relevance and bonds)
        attn_out = self.attention_module.compute(vectors)
        
        # 4. Apply Feed-Forward Network (add non-linearity and computation capacity)
        output_vectors = self.feed_forward_module.compute(attn_out)
        
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "output_vectors": output_vectors
        }

class RelationalPerspectiveNode:
    """
    An upgraded node that encodes relationships between concepts 
    rather than just the concepts themselves, based on a specific perspective.
    """
    def __init__(self, perspective, vocab, embedding_dim):
        self.perspective = perspective
        self.vocab = vocab
        self.embedding = EmbeddingMap(len(vocab), embedding_dim)
        self.attention = SelfAttention(embedding_dim)

        # relation_encoder maps concatenated (vec_a, vec_b) back to embedding_dim
        # Using a simple weight matrix (dim*2, dim) and bias (dim,)
        limit = np.sqrt(6 / (embedding_dim * 2 + embedding_dim))
        self.W_rel = np.random.uniform(-limit, limit, (embedding_dim * 2, embedding_dim))
        self.b_rel = np.zeros(embedding_dim)
        
        self.relation_memory = {}   # stores seen relationships

    def _encode_concept(self, text):
        combined = f"{self.perspective} {text}"
        tokens = encode(combined, self.vocab)
        embedded = self.embedding.embed_sequence(tokens)
        attended = self.attention.compute(embedded)
        # Mean across the sequence length (axis 0) returning (dim,)
        return np.mean(attended, axis=0)

    def process(self, concept_a, concept_b=None):
        # Encode concept A
        vec_a = self._encode_concept(concept_a)

        if concept_b is None:
            # Single concept mode — retrieve stored relationships
            return self.recall_relations(vec_a)

        # Encode concept B
        vec_b = self._encode_concept(concept_b)

        # Encode the RELATIONSHIP — not the concepts
        pair = np.concatenate([vec_a, vec_b])           # (dim*2,)
        relation = np.dot(pair, self.W_rel) + self.b_rel  # (dim,)

        # Store it
        key = f"{concept_a}→{concept_b}"
        self.relation_memory[key] = relation

        return relation   # output: the BRIDGE not the endpoints

    def recall_relations(self, query_vec):
        # Find stored relations most similar to this query
        if not self.relation_memory:
            return query_vec

        best_score = -float('inf')
        best_relation = query_vec

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            query_norm = 1e-8

        for key, stored_vec in self.relation_memory.items():
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm == 0:
                stored_norm = 1e-8
            
            # Cosine similarity
            score = np.dot(query_vec, stored_vec) / (query_norm * stored_norm)
            if score > best_score:
                best_score = score
                best_relation = stored_vec

        return best_relation
