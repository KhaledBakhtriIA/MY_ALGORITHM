from tokenizer import build_vocab, encode, decode
from embeddings import EmbeddingMap
from attention import SelfAttention
from node import PerspectiveNode
from run_nodes import run_all_nodes
from synthesizer import synthesize
from output_layer import OutputProjection, softmax
from loss import cross_entropy_loss
from backprop import backpropagate_step_sequence
from ffn import FeedForward
import math
import numpy as np
import random

def calculate_dim_for_fanout(vocab_size, num_nodes, overlap_factor=0.3):
    """
    Method 4: The 6-Node Multiplier Rule
    overlap_factor: how much nodes are allowed to share dimensions (0.0–1.0)
    lower = more specialized nodes, needs higher dim
    """
    # Base dim from vocab
    base_dim = 2 ** math.ceil(math.log2(vocab_size ** 0.5))
    
    # Each node needs headroom — but they share the embedding space
    # so we don't multiply by full num_nodes
    effective_nodes = num_nodes * (1 - overlap_factor)
    
    raw_dim = base_dim * effective_nodes
    
    # Round up to nearest power of 2 (hardware efficiency)
    final_dim = 2 ** math.ceil(math.log2(raw_dim))
    
    return int(final_dim)

def run_training_loop():
    print("1. Initializing Training Environment...")
    
    test_dataset = [
        # Simple definitions
        ("what is water",           "water is a clear liquid made of hydrogen and oxygen"),
        ("what is fire",            "fire is a chemical reaction that produces heat and light"),
        ("what is air",             "air is a mixture of gases that surrounds the earth"),
        ("what is soil",            "soil is the top layer of earth where plants grow"),

        # Data / Tech related (close to your training target)
        ("what is data",            "data means facts and statistics about everything"),
        ("what is code",            "code is a set of instructions written for computers"),
        ("what is a network",       "a network is a system of connected computers and devices"),
        ("what is an algorithm",    "an algorithm is a step by step method to solve a problem"),
        ("what is machine learning","machine learning is when computers learn from data automatically"),
        
        # Unrelated Concepts (Pushing concepts apart)
        ("what separates data from objects",  "data is abstract while objects are physical"),
        ("is a dog the same as data",         "no a dog is living while data is information"),

        # Science
        ("what is gravity",         "gravity is a force that pulls objects toward each other"),
        ("what is energy",          "energy is the ability to do work or cause change"),
        ("what is a cell",          "a cell is the smallest unit of life in living things"),
        ("what is DNA",             "DNA is a molecule that carries genetic information in living things"),

        # General knowledge
        ("what is a language",      "a language is a system of words used to communicate"),
        ("what is time",            "time is the ongoing sequence of events from past to future"),
        ("what is money",           "money is a medium used to exchange goods and services"),
        ("what is health",          "health is the state of being free from illness and injury"),

        # Short and simple
        ("what is a dog",           "a dog is a domestic animal kept as a pet or for work"),
        ("what is a book",          "a book is a set of written pages bound together"),
        ("what is a school",        "a school is a place where people go to learn and study"),
    ]
    
    # Setup specific perspectives — these MUST match the 6 nodes defined in
    # main.py so the trained weights line up with the inference pipeline.
    prompts = {
        "Optimist": "Focus on the positive potential and opportunities.",
        "Pessimist": "Identify the risks, flaws, and constraints.",
        "Analyst": "Look purely at facts, data, and logical sequences.",
        "Philosopher": "Consider the ethical and deeper meaning implications.",
        "Innovator": "Think outside the box and suggest novel paradigms.",
        "Pragmatist": "Focus on actionable, realistic execution steps."
    }

    # Vocabulary setup
    corpus = [item for pair in test_dataset for item in pair]
    vocab = build_vocab(corpus)
    vocab_size = len(vocab)
    embedding_dim = calculate_dim_for_fanout(vocab_size=vocab_size, num_nodes=6)
    print(f"   -> Setup: Vocab size {vocab_size}, Calculated dynamic embedding dimension: {embedding_dim}")
    
    # Initialize Parameters
    embedding_map = EmbeddingMap(vocab_size, embedding_dim)
    output_projection = OutputProjection(embedding_dim, vocab_size)
    shared_attention = SelfAttention(embedding_dim)
    shared_ffn = FeedForward(embedding_dim)
    
    nodes = []
    for name, prompt in prompts.items():
        nodes.append(PerspectiveNode(name, prompt, vocab, embedding_map, shared_attention, shared_ffn))

    epochs = 800
    base_learning_rate = 0.08
    
    print(f"Starting {epochs} training epochs on dataset of size {len(test_dataset)}...\n")

    for epoch in range(epochs):
        learning_rate = base_learning_rate * (0.5 * (1 + math.cos(math.pi * epoch / epochs)))

        # Shuffle dataset each epoch
        random.shuffle(test_dataset)
        
        epoch_loss = 0.0
        
        for input_query, target_response in test_dataset:
            target_ids = encode(target_response, vocab)
            
            # To ensure the network has enough capacity/length to generate the entire
            # target sequence, we pad the input query with spaces until it's long enough
            # to match the target response token count.
            padded_query = input_query
            while len(encode(padded_query, vocab)) < len(target_ids):
                padded_query += " " + input_query.split()[-1] # repeat last word for padding
                
            input_ids = encode(padded_query, vocab)
            
            # --- FORWARD PASS ---
            node_results = run_all_nodes(nodes, padded_query)
            final_vectors = synthesize(node_results)
            
            sequence_logits = []
            for vec in final_vectors:
                logits = output_projection.project(vec)
                sequence_logits.append(logits)
                
            # --- LOSS CALCULATION ---
            sequence_probs = [softmax(logits) for logits in sequence_logits]
            loss = cross_entropy_loss(sequence_probs, target_ids)
            epoch_loss += loss
            
            # --- BACKWARD PASS ---
            steps = min(len(sequence_logits), len(target_ids))
            
            # Process the entire sequence backwards uniformly
            seq_vecs = final_vectors[:steps]
            seq_logits = sequence_logits[:steps]
            seq_t_ids = target_ids[:steps]
            
            seq_in_ids = []
            for i in range(steps):
                seq_in_ids.append(input_ids[min(i, len(input_ids) - 1)])

            if epoch == 0:
                first_token = input_ids[0]
                print(f"[Weight Update Check - Epoch 0]")
                print(f"  Attention W_q[0][0] before: {shared_attention.W_q[0][0]:.8f}")
                print(f"  Embedding[{first_token}][0] before: {embedding_map.matrix[first_token][0]:.8f}")

            sum_d_emb = backpropagate_step_sequence(
                output_projection=output_projection,
                embedding_map=embedding_map,
                nodes=nodes,
                input_token_ids=seq_in_ids,
                vectors=seq_vecs,
                logits_seq=seq_logits,
                target_ids=seq_t_ids,
                learning_rate=learning_rate
            )
            
            if epoch == 0:
                print(f"  Gradient on Embedding (dW) for token 0, dim 0: {sum_d_emb[0][0]:.8f}")
                print(f"  Avg absolute Gradient on Embedding (dW) for seq: {np.mean(np.abs(sum_d_emb)):.8f}")
                print(f"  Attention W_q[0][0] after:  {shared_attention.W_q[0][0]:.8f}")
                print(f"  Embedding[{first_token}][0] after:  {embedding_map.matrix[first_token][0]:.8f}")
            
        # --- LOGGING ---
        avg_loss = epoch_loss / len(test_dataset)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")
            # Generate a test sample from a fixed question
            test_query = "what is data"
            test_res = run_all_nodes(nodes, test_query)
            test_vecs = synthesize(test_res)
            output_text = output_projection.generate_tokens(test_vecs, vocab, temperature=0.7, rep_penalty=1.5)
            print(f"  Q: '{test_query}' -> {output_text}")

    print("\nTraining Complete! Saving model weights to disk...")

    # Save weights
    import json
    import os
    
    # 1. Save Vocabulary MAP
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 2. Save Embedding Map
    np.save("embedding_weights.npy", np.array(embedding_map.matrix))

    # 3. Save Attention Module
    np.save("attention_W_q.npy", shared_attention.W_q)
    np.save("attention_W_k.npy", shared_attention.W_k)
    np.save("attention_W_v.npy", shared_attention.W_v)

    # 4. Save Output Projection
    np.save("output_matrix.npy", np.array(output_projection.matrix))

    print("Weights successfully saved!")
    
if __name__ == "__main__":
    run_training_loop()