import random

# A mock dataset of tuples (input_sequence, target_sequence)
original_dataset = [
    ("what is data", "data is facts and information"),
    ("what is fire", "fire is heat and light"),
]

# Define a dictionary of semantic synonyms
synonyms = {
    "data": ["information", "metrics", "statistics"],
    "is": ["means", "represents", "equals"],
    "what": ["explain", "define", "describe"],
    "fire": ["flames", "combustion"],
}

def augment_dataset(dataset, multiplier=5):
    augmented = []
    
    for _ in range(multiplier):
        for seq_in, seq_out in dataset:
            # 1. Synonym Replacement
            words_in = seq_in.split()
            words_out = seq_out.split()
            
            new_in = [random.choice(synonyms.get(w, [w])) for w in words_in]
            new_out = [random.choice(synonyms.get(w, [w])) for w in words_out]
            
            # 2. Random Dropping (Simulating noise to prevent rigid memorization)
            if random.random() < 0.2 and len(new_in) > 2:
                drop_idx = random.randint(0, len(new_in) - 1)
                new_in.pop(drop_idx)
                
            augmented.append((" ".join(new_in), " ".join(new_out)))
            
    return augmented

# This turns 2 examples into 10 examples mathematically distinct but logically identical
expanded_data = augment_dataset(original_dataset, multiplier=5)

for i, (q, a) in enumerate(expanded_data):
    print(f"Example {i+1}: '{q}' -> '{a}'")
