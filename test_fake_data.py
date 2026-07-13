import json
from tokenizer import build_vocab
from embeddings import EmbeddingMap
from attention import SelfAttention
from node import PerspectiveNode
from run_nodes import run_all_nodes
from synthesizer import synthesize
from output_layer import OutputProjection

def test_neural_network():
    print("1. Loading fake data...")
    fake_data = [
      {"sender": "Customer", "message": "Hi, I want to know if the laptop is still available?", "intent": "product inquiry"},
      {"sender": "Agent", "message": "Yes, it’s available. Would you like me to share the current price?", "intent": "confirmation"},
      {"sender": "Customer", "message": "Sure, please tell me the price.", "intent": "price request"},
      {"sender": "Agent", "message": "The laptop is currently $999, with a 10% discount this week.", "intent": "price info"},
      {"sender": "Customer", "message": "Great, I’ll place the order today.", "intent": "purchase decision"}
    ]
    
    # Extract messages to build the vocabulary
    corpus = [item["message"] for item in fake_data]
    
    vocab = build_vocab(corpus)
    vocab_size = len(vocab)
    embedding_dim = 16
    
    print(f"   -> Vocab size: {vocab_size}, Embedding dim: {embedding_dim}")

    # Shared Core Intelligence and Maps
    embedding_map = EmbeddingMap(vocab_size, embedding_dim)
    attention_module = SelfAttention(embedding_dim)
    output_projection = OutputProjection(embedding_dim, vocab_size)

    # Define some perspectives (e.g. focused on different aspects of customer service)
    prompts = {
        "CustomerService": "Focus on helpfulness and providing accurate information.",
        "Sales": "Identify purchase intent and drive conversions.",
        "Analyst": "Analyze the customer intent and extract key details."
    }

    print("2. Setting up perspective nodes...")
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

    print("3. Testing the network on the fake data sequences...")
    for idx, data in enumerate(fake_data):
        input_query = data["message"]
        sender = data["sender"]
        intent = data["intent"]
        
        print(f"\n--- Testing Message {idx + 1} ---")
        print(f"Sender: {sender}, Intent: {intent}")
        print(f"Input Query: '{input_query}'")
        
        # The Fan-Out
        node_results = run_all_nodes(nodes, input_query)

        # The Convergence
        final_output = synthesize(node_results)

        # The Projection
        generated_text = output_projection.generate_tokens(final_output, vocab)
        
        print(f"Generated Output Context: '{generated_text}'")

    print("\n--- TEST COMPLETE ---")

if __name__ == "__main__":
    test_neural_network()
