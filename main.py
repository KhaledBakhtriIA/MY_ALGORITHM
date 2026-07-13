from tokenizer import build_vocab
from embeddings import EmbeddingMap
from attention import SelfAttention
from node import PerspectiveNode
from run_nodes import run_all_nodes
from synthesizer import synthesize
from output_layer import OutputProjection

import json
import os
import math
import numpy as np

def main():
    print("1. Initializing pipeline & setting up shared modules...")
    
    # Try to load existing model
    trained_model_exists = all(map(os.path.exists, [
        "vocab.json", 
        "embedding_weights.npy", 
        "attention_W_q.npy", 
        "attention_W_k.npy", 
        "attention_W_v.npy", 
        "output_matrix.npy"
    ]))
    
    if trained_model_exists:
        print("  -> Loading pre-trained weights from disk...")
        with open("vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
            
        vocab_size = len(vocab)
        
        embeddings_arr = np.load("embedding_weights.npy").tolist()
        embedding_dim = len(embeddings_arr[0])
        
        embedding_map = EmbeddingMap(vocab_size, embedding_dim)
        embedding_map.matrix = embeddings_arr
        
        attention_module = SelfAttention(embedding_dim)
        attention_module.W_q = np.load("attention_W_q.npy")
        attention_module.W_k = np.load("attention_W_k.npy")
        attention_module.W_v = np.load("attention_W_v.npy")
        
        output_projection = OutputProjection(embedding_dim, vocab_size)
        output_projection.matrix = np.load("output_matrix.npy").tolist()
        
    else:
        print("  -> NO PRE-TRAINED MODEL FOUND! Using random initialization...")
        # Simple training corpus to populate the vocabulary for testing
        corpus = [
            "this is a sample training line",
            "another line for vocabulary building",
            "we need enough words to make embeddings work",
            "what is the meaning of data and everything",
            "artificial intelligence algorithms can learn complex patterns",
            "deep neural networks require massive amounts of structured text",
            "the universe is vast and expands constantly through space and time",
            "philosophical thought questions the very nature of human existence",
            "pragmatic approaches focus on actionable realistic execution steps",
            "optimists believe in positive potential and future opportunities",
            "pessimists identify risks flaws constraints and inevitable failures",
            "data analysts look purely at facts numbers logical sequences",
            "innovators think outside the box and suggest novel paradigms",
            "machine learning models process vectors matrices and continuous floating point numbers",
            "gradient descent minimizes the loss function over multiple training epochs",
            "cross entropy measures the difference between probability distributions",
            "attention mechanisms calculate the relevance of different tokens in a sequence",
            "synthesizing multiple perspectives yields a more balanced and robust conclusion",
            "the quick brown fox jumps over the lazy dog",
            "natural language processing bridges the gap between computers and human communication",
            "understanding context and semantics is crucial for accurate translation",
            "software engineering involves writing testing and maintaining code",
            "open source repositories host millions of projects worldwide",
            "the internet connects billions of devices allowing seamless information exchange",
            "cybersecurity protects networks systems and programs from digital attacks"
        ]
        input_query = "what is the meaning of data?"
        
        vocab = build_vocab(corpus + [input_query])
        vocab_size = len(vocab)
        
        # Recalculate based on fan-out algorithm
        base_dim = 2 ** math.ceil(math.log2(vocab_size ** 0.5))
        effective_nodes = 6 * (1 - 0.3)
        raw_dim = base_dim * effective_nodes
        embedding_dim = int(2 ** math.ceil(math.log2(raw_dim)))
        
        # Shared Core Intelligence and Maps
        embedding_map = EmbeddingMap(vocab_size, embedding_dim)
        attention_module = SelfAttention(embedding_dim)
        output_projection = OutputProjection(embedding_dim, vocab_size)
    
    input_query = "what is data"
    print(f"   -> Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")

    # Define the 6 unique system prompts (perspectives) 
    print("2. Defining the 6 perspective nodes...")
    prompts = {
        "Optimist": "Focus on the positive potential and opportunities.",
        "Pessimist": "Identify the risks, flaws, and constraints.",
        "Analyst": "Look purely at facts, data, and logical sequences.",
        "Philosopher": "Consider the ethical and deeper meaning implications.",
        "Innovator": "Think outside the box and suggest novel paradigms.",
        "Pragmatist": "Focus on actionable, realistic execution steps."
    }

    nodes = []
    for name, prompt in prompts.items():
        node = PerspectiveNode(
            name=name,
            system_prompt=prompt,
            vocab=vocab,
            embedding_map=embedding_map,
            attention_module=attention_module
        )
        nodes.append(node)

    # The Fan-Out
    print(f"3. The Fan-Out: Running 6 perspectives simultaneously on input query: '{input_query}'")
    node_results = run_all_nodes(nodes, input_query)

    for res in node_results:
        vecs = res.get('output_vectors', [])
        print(f"   -> Node [{res['name']}] completed. Generated {len(vecs)} position vectors.")

    # The Convergence
    print("4. The Convergence: Synthesizing vectors into final bonded output...")
    final_output = synthesize(node_results)

    print(f"5. The Projection: Generating contextual response from synthesized representation layers...")
    generated_text = output_projection.generate_tokens(final_output, vocab, temperature=0.7, rep_penalty=1.5)
    
    print("\n--- FORWARD PASS COMPLETE ---")
    print(f"Input Query:     '{input_query}'")
    print(f"Final Model Output: '{generated_text}'\n")

if __name__ == "__main__":
    main()