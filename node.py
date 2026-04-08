from tokenizer import encode
from embeddings import EmbeddingMap
from attention import SelfAttention

class PerspectiveNode:
    """
    A single perspective engine that takes a system prompt and data, 
    processes them through the embedding and attention modules, 
    and returns a unique reading of the data.
    """
    def __init__(self, name, system_prompt, vocab, embedding_map, attention_module):
        self.name = name
        self.system_prompt = system_prompt
        self.vocab = vocab
        self.embedding_map = embedding_map
        self.attention_module = attention_module

    def process(self, data):
        """
        Runs the combined system prompt and input data through 
        tokenization, embedding, and attention to generate a perspective output.
        """
        # Combine system prompt with input data
        combined_text = f"{self.system_prompt} {data}"
        
        # 1. Tokenize (convert to integers)
        token_ids = encode(combined_text, self.vocab)
        
        # 2. Embed (convert integers to vectors)
        vectors = self.embedding_map.embed_sequence(token_ids)
        
        # 3. Apply Attention (calculate relevance and bonds)
        output_vectors = self.attention_module.compute(vectors)
        
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "output_vectors": output_vectors
        }