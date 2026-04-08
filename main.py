from tokenizer import build_vocab
from embeddings import EmbeddingMap
from attention import SelfAttention
from node import PerspectiveNode
from run_nodes import run_all_nodes
from synthesizer import synthesize
from output_layer import OutputProjection

def main():
    print("1. Initializing pipeline & setting up shared modules...")
    
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
    embedding_dim = 16
    
    print(f"   -> Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")

    # Shared Core Intelligence and Maps
    embedding_map = EmbeddingMap(vocab_size, embedding_dim)
    attention_module = SelfAttention(embedding_dim)
    output_projection = OutputProjection(embedding_dim, vocab_size)

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
    generated_text = output_projection.generate_tokens(final_output, vocab)
    
    print("\n--- FORWARD PASS COMPLETE ---")
    print(f"Input Query:     '{input_query}'")
    print(f"Final Model Output: '{generated_text}'\n")

if __name__ == "__main__":
    main()