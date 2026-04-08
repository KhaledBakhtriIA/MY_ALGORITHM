from tokenizer import build_vocab, encode, decode
from embeddings import EmbeddingMap
from attention import SelfAttention
from node import PerspectiveNode
from run_nodes import run_all_nodes
from synthesizer import synthesize
from output_layer import OutputProjection
from loss import sequence_loss
from backprop import backpropagate_step

def run_training_loop():
    print("1. Initializing Training Environment...")
    
    # Query and expected response
    input_query = "what is the meaning of data"
    # Keeping target length identical to the synthesized sequence length (Prompt (1) + Query (6) = 7)
    target_response = "data means facts and statistics about everything"
    
    # Vocabulary setup
    corpus = [
        input_query,
        target_response,
        "padding words to make vocabulary larger",
        "more text for embeddings",
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
    vocab = build_vocab(corpus)
    vocab_size = len(vocab)
    embedding_dim = 16
    
    target_ids = encode(target_response, vocab)
    input_ids = encode(input_query, vocab)
    
    # Initialize Parameters
    embedding_map = EmbeddingMap(vocab_size, embedding_dim)
    attention_module = SelfAttention(embedding_dim)
    output_projection = OutputProjection(embedding_dim, vocab_size)
    
    # Setup specific perspectives
    prompts = {
        "Optimist": "positive",
        "Pessimist": "negative",
        "Analyst": "logic",
        "Philosopher": "ethics"
    }
    
    nodes = []
    for name, prompt in prompts.items():
        nodes.append(PerspectiveNode(name, prompt, vocab, embedding_map, attention_module))

    epochs = 200
    learning_rate = 0.05
    
    print(f"\nTarget Response: '{target_response}'")
    print(f"Starting {epochs} training epochs...\n")

    for epoch in range(epochs):
        # --- FORWARD PASS ---
        node_results = run_all_nodes(nodes, input_query)
        final_vectors = synthesize(node_results)
        
        sequence_logits = []
        for vec in final_vectors:
            logits = output_projection.project(vec)
            sequence_logits.append(logits)
            
        # --- LOSS CALCULATION ---
        loss = sequence_loss(sequence_logits, target_ids)
        
        # --- BACKWARD PASS ---
        valid_steps = min(len(sequence_logits), len(target_ids))
        for i in range(valid_steps):
            vec = final_vectors[i]
            logits = sequence_logits[i]
            t_id = target_ids[i]
            
            # Determine which input token embedding gets updated 
            # (In a full scale architecture, attention passes gradients exactly to their originating source targets.
            # Here we map directly to the corresponding input sequence index as an educational abstraction representation).
            in_id = input_ids[i] if i < len(input_ids) else 0
            
            backpropagate_step(
                output_projection=output_projection,
                embedding_map=embedding_map,
                input_token_id=in_id,
                vector=vec,
                logits=logits,
                target_id=t_id,
                learning_rate=learning_rate
            )
            
        # --- LOGGING ---
        if epoch % 20 == 0 or epoch == epochs - 1:
            # We use a very low temperature here just to observe the most confident prediction the model has learned
            output_text = output_projection.generate_tokens(final_vectors, vocab, temperature=0.1)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Prediction: {output_text}")

if __name__ == "__main__":
    run_training_loop()