import torch
import torch.nn as nn
import torch.optim as optim
from model_pt import CustomTransformer

# Mocking the constants for testing based on your prior values
VOCAB_SIZE = 131
EMBEDDING_DIM = 128
EPOCHS = 800

# Select device: NVIDIA GPU if PyTorch finds it, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# Initialize model directly on the hardware (GPU/CPU)
model = CustomTransformer(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM).to(device)

# Instead of manual Mean Squared Error derivative, PyTorch calculates CrossEntropy over Softmax distributions
criterion = nn.CrossEntropyLoss()

# Adam Optimizer completely replaces manual W = W - lr * dW inside backprop.py
# It automatically speeds up/slows down learning rates per weight (Momentum)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# We must train it predicting exactly the *next* word!
# E.g. "what" -> "is", "what is" -> "data"
full_sequence = [5, 8, 12, 45, 99, 100]  # Let's say: [What, is, data, ?, <end>]
inputs = torch.tensor([full_sequence[:-1]], dtype=torch.long).to(device)  # [What, is, data, ?]
targets = torch.tensor([full_sequence[1:]], dtype=torch.long).to(device)  # [is, data, ?, <end>]

print("Starting Autoregressive Training (LLM style)...")
for epoch in range(EPOCHS):
    model.train() # Set to train mode
    
    # 1. Clear gradients from the previous step
    optimizer.zero_grad()
    
    # 2. Forward pass: compute predictions
    # This automatically builds the computational graph in the background
    logits = model(inputs) 
    
    # Reshape for CrossEntropyLoss: (Batch * SeqLen, Vocab_size)
    logits_flat = logits.view(-1, VOCAB_SIZE)
    targets_flat = targets.view(-1)
    
    # 3. Calculate loss
    loss = criterion(logits_flat, targets_flat)
    
    # 4. Backward pass: Auto-calculates all gradients! (Completely replaces backprop.py)
    loss.backward()
    
    # 5. Update Weights step using Adam.
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

print("\n--- GENERATION TEST: WATCH YOUR AI WRITE ---")
model.eval() # Turn off training settings (like Dropout, if we had it)
with torch.no_grad(): # Don't record math gradients during pure generation
    # Provide a "seed" prompt to start generation: [What, is]
    prompt_sequence = [5, 8] 
    generated = prompt_sequence.copy()
    
    print(f"Input Prompt tokens: {prompt_sequence}")
    
    # Generate the next 4 tokens autonomously
    for i in range(4):
        # Convert our growing list into a tensor for the model
        current_input = torch.tensor([generated], dtype=torch.long).to(device)
        
        # Look at the whole sentence, predict everything
        predictions = model(current_input)
        
        # Grab the logits for the VERY LAST word in the sequence it just analyzed
        next_word_logits = predictions[0, -1, :] 
        
        # Which mathematical token had the highest probability? (Argmax)
        next_token = torch.argmax(next_word_logits).item()
        
        # Append it to the sequence and feed it back into itself!
        generated.append(next_token)
        print(f"Step {i+1}: Generated token ID [{next_token}] -> Current Sentence: {generated}")
        
    print("\nFinal Generated Token Sequence:", generated)
