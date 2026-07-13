from node import RelationalPerspectiveNode
from tokenizer import build_vocab

def run_relational_test():
    print("1. Initializing Relational Training Environment...")
    
    # 1. Setup vocabulary containing our concepts and prompts
    corpus = [
        "data statistics insight opportunity noise truth tool",
        "Analyst Innovator Optimist Pessimist Philosopher Pragmatist",
        "positive negative logic ethics",
        "what is the meaning of data"
    ]
    vocab = build_vocab(corpus)
    embedding_dim = 16
    
    # 2. Define perspectives and their learned relationships for "data"
    perspectives = {
        "Analyst": ("data", "statistics"),
        "Innovator": ("data", "insight"),
        "Optimist": ("data", "opportunity"),
        "Pessimist": ("data", "noise"),
        "Philosopher": ("data", "truth"),
        "Pragmatist": ("data", "tool")
    }
    
    nodes = {}
    for name in perspectives.keys():
        nodes[name] = RelationalPerspectiveNode(name, vocab, embedding_dim)
        
    print("\n2. Training Relationships...")
    for name, (concept_a, concept_b) in perspectives.items():
        node = nodes[name]
        # Providing both concepts automatically encodes and stores the relationship
        relation_vector = node.process(concept_a, concept_b)
        print(f"[{name:12s}] Encoded and stored relationship: {concept_a} → {concept_b}")
        
    print("\n3. Testing / Recalling Relationships...")
    # 3. Test recalling the stored relation by providing only the primary concept
    for name in perspectives.keys():
        node = nodes[name]
        # Querying with just "data" triggers the `recall_relations` path
        recall_vec = node.process("data")
        
        # Verify that a vector was recalled
        print(f"[{name:12s}] Recalled relation vector for 'data' (Shape: {recall_vec.shape})")
        
    print("\n4. Testing Analogy / Unseen Word...")
    for name in perspectives.keys():
        node = nodes[name]
        # Querying with an unseen word or phrase just calculates the closest match
        recall_vec_unseen = node.process("information")
        print(f"[{name:12s}] Queried 'information' -> fell back to closest stored relation (Cosine Sim). Shape: {recall_vec_unseen.shape}")

if __name__ == "__main__":
    run_relational_test()
