from tokenizer import build_vocab
from embeddings import EmbeddingMap
from attention import SelfAttention
from node import PerspectiveNode
from run_nodes import run_all_nodes
from synthesizer import synthesize

def main():
    print("1. Initializing pipeline & setting up shared modules...")
    
    # Simple training corpus to populate the vocabulary for testing
    corpus = [
        "this is a sample training line",
        "another line for vocabulary building",
        "we need enough words to make embeddings work",
        "what is the meaning of data and everything"
    ]
    input_query = "what is the meaning of data?"
    
    vocab = build_vocab(corpus + [input_query])
    vocab_size = len(vocab)
    embedding_dim = 16
    
    print(f"   -> Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")

    # Shared Core Intelligence and Maps
    embedding_map = EmbeddingMap(vocab_size, embedding_dim)
    attention_module = SelfAttention(embedding_dim)

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

    print(f"5. Pipeline Complete. Final representation shape: {len(final_output)} tokens * {embedding_dim} dim vectors.")

if __name__ == "__main__":
    main()