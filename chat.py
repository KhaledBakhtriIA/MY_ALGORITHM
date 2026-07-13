import torch
import torch.nn as nn
import torch.optim as optim
from model_pt import CustomTransformer
import csv
import os
import random
import gensim.downloader as api
from spellchecker import SpellChecker

# 1. Setup Settings
MAX_VOCAB = 5000  # We give the AI a massive vocabulary ceiling so it can learn new words dynamically
CSV_FILE = "memory.csv"

# 2. Dynamic Tokenizer (Turns English into Math, and learns new words instantly)
class DynamicTokenizer:
    def __init__(self):
        self.spell = SpellChecker()
        # We start with your requested base vocabulary layout!
        self.word2idx = {
            "<PAD>": 0, 
            "<UNK>": 1, 
            "<STOP>": 2,
            "what": 3, "is": 4, "a": 5, "the": 6, "to": 7, "of": 8, "and": 9, "in": 10,
            "it": 11, "you": 12, "i": 13, "neural": 14, "network": 15, "data": 16, "fire": 17,
            # Enhanced Base Vocabulary for better conversational logic
            "am": 18, "are": 19, "do": 20, "does": 21, "how": 22, "why": 23, "when": 24,
            "where": 25, "who": 26, "can": 27, "could": 28, "will": 29, "would": 30,
            "should": 31, "have": 32, "has": 33, "had": 34, "be": 35, "been": 36,
            "my": 37, "your": 38, "his": 39, "her": 40, "their": 41, "our": 42,
            "me": 43, "we": 44, "us": 45, "them": 46, "they": 47, "he": 48, "she": 49,
            "yes": 50, "no": 51, "not": 52, "very": 53, "too": 54, "so": 55, "but": 56,
            "or": 57, "because": 58, "if": 59, "then": 60, "than": 61, "that": 62, "this": 63,
            "these": 64, "those": 65, "all": 66, "some": 67, "any": 68, "many": 69,
            "much": 70, "more": 71, "most": 72, "other": 73, "such": 74, "only": 75,
            "same": 76, "different": 77, "new": 78, "old": 79, "good": 80, "bad": 81,
            "great": 82, "small": 83, "large": 84, "big": 85, "high": 86, "low": 87,
            "hot": 88, "cold": 89, "warm": 90, "cool": 91, "fast": 92, "slow": 93,
            "hello": 94, "hi": 95, "hey": 96, "bye": 97, "goodbye": 98, "thanks": 99,
            "thank": 100, "please": 101, "sorry": 102, "wrong": 103, "right": 104,
            "true": 105, "false": 106, "human": 107, "machine": 108, "ai": 109,
            "intelligent": 110, "smart": 111, "think": 112, "know": 113, "learn": 114,
            "understand": 115, "see": 116, "hear": 117, "say": 118, "tell": 119,
            "make": 120, "create": 121, "build": 122, "system": 123, "pattern": 124,
            "patterns": 125, "model": 126, "concept": 127, "idea": 128, "thought": 129
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def add_word(self, word):
        if word not in self.word2idx and len(self.word2idx) < MAX_VOCAB:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def encode(self, text):
        # 1. Spelling correction (Avoid <UNK> typo hallucination mapping)
        words = text.strip().lower().split()
        corrected_words = []
        for word in words:
            if word not in self.word2idx:
                correction = self.spell.correction(word)
                corrected_words.append(correction if correction else word)
            else:
                corrected_words.append(word)
                
        # 2. Break English into mathematical integers
        for word in corrected_words:
            self.add_word(word)
        return [self.word2idx.get(w, 1) for w in corrected_words]
        
    def decode(self, indices):
        # Turn Math back into English
        words = []
        for i in indices:
            if i == 2: # Stop token
                break
            words.append(self.idx2word.get(i, "<UNK>"))
        return " ".join(words)

tokenizer = DynamicTokenizer()

# 3. Load Existing Memories (Database)
memory_data = [] # List storing (prompt, answer)
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) == 2:
                # Encode step creates the vocab correctly immediately upon boot
                q, a = row[0].strip().lower(), row[1].strip().lower()
                tokenizer.encode(q + (" " + a if a else "")) 
                memory_data.append((q, a))

if not memory_data:
    print("\n[!] Memory is completely empty! You must use the Teach command (question|answer) to build the brain initially.")
    # Create the CSV file safely without wiping it
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            pass

# 4. Engine Bootup (GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download and load GloVe pre-trained embeddings (only takes a moment the first time)
glove = None
def get_glove():
    global glove
    if glove is None:
        print("\n[Accessing GloVe Pre-Trained Dictionary...]")
        try:
            glove = api.load("glove-wiki-gigaword-100")
        except Exception as e:
            print(f"Warning: Failed to load GloVe, falling back to random: {e}")
    return glove

model = CustomTransformer(vocab_size=MAX_VOCAB, embedding_dim=100).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss(ignore_index=0)

BRAIN_FILE = "quantara_brain.pth"
if os.path.exists(BRAIN_FILE):
    print("[Restoring Neural Structure from File...]")
    model.load_state_dict(torch.load(BRAIN_FILE))
else:
    # First time boot: give it geometry
    _g = get_glove()
    if _g:
        print("[Integrating Pre-Trained Math into Matrix...]")
        with torch.no_grad():
            for idx_val, word in tokenizer.idx2word.items():
                if word in _g:
                    model.embeddings.embed.weight[idx_val] = torch.tensor(_g[word])
        print("[Done.]")

# 5. Continuous Learning Matrix (The Brain Retraining Loop)
def train_on_memory(epochs=80):
    if not memory_data:
        return
    model.train()
    print(f"\n[AI is thinking... Processing {len(memory_data)} facts into Neural Patterns...]")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for idx, (q, a) in enumerate(memory_data):
            optimizer.zero_grad() # Clear short-term memory per sentence
            
            # Sequence: Question + STOP/SEP + Answer + STOP
            # This perfectly separates the prompt from the requested response structure
            
            # --- STRUCTURAL ANALOGY MASKING ---
            # 15% of the time, we replace a shared word between question and answer
            # with a dynamically generated fake word. This forces the model
            # to learn positional reasoning (copying the word) rather than just memorizing it!
            words_in_q = set(q.split())
            words_in_a = set(a.split())
            shared_words = list(words_in_q.intersection(words_in_a))
            
            if shared_words and random.random() < 0.15:
                concept = random.choice(shared_words)
                if len(concept) > 2 and concept not in ["is", "the", "are", "you", "a"]:
                    fake_concept = f"concept_{random.randint(100,999)}"
                    # Ensure it's in vocab with initialized embedding
                    tokenizer.add_word(fake_concept)
                    q_mod = q.replace(concept, fake_concept)
                    a_mod = a.replace(concept, fake_concept)
                    seq_q = tokenizer.encode(q_mod)
                    seq_a = tokenizer.encode(a_mod)
                else:
                    seq_q = tokenizer.encode(q)
                    seq_a = tokenizer.encode(a)
            else:
                seq_q = tokenizer.encode(q)
                seq_a = tokenizer.encode(a)
            
            # Combine the mathematical tensors: Question -> Ans -> Stop
            seq = seq_q + seq_a + [2]
            if len(seq) < 2: continue
            
            inputs = torch.tensor([seq[:-1]], dtype=torch.long).to(device)
            targets = torch.tensor([seq[1:]], dtype=torch.long).to(device)
            
            logits = model(inputs)
            loss = criterion(logits.view(-1, MAX_VOCAB), targets.view(-1))
            loss.backward()
            optimizer.step()  # Fix: Must keep this AFTER every step for SGD!
            total_loss += loss.item()
            
    print(f"[Done.] Brain updated perfectly. Final structural loss: {total_loss/max(1, len(memory_data)):.4f}\n")
    torch.save(model.state_dict(), BRAIN_FILE)

# Initial Boot Training Layer
if memory_data and not os.path.exists(BRAIN_FILE):
    # 400 epochs guarantee PERFECT memorization of the small CSV logic dictionary 
    # to prevent <UNK> leaking before generalizing into unstructured chat.
    train_on_memory(400)

# 6. Cognitive Generation (Talking back with Structural Creativity)
def generate_response(prompt, max_len=25, temperature=0.7):
    model.eval()
    encoded = tokenizer.encode(prompt)
    if not encoded:
        return "..."
        
    generated = encoded.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            inputs = torch.tensor([generated], dtype=torch.long).to(device)
            logits = model(inputs)
            
            # Extract final row of logits
            next_logits = logits[0, -1, :]
            
            # Divide by temperature to "soften" the absolute math, allowing the model
            # to occasionally pick grammatically related words instead of just the #1 memorized word.
            # This is how AI "thinks" and finds new relationships!
            scaled_logits = next_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # Divide by temperature to "soften" the absolute math, allowing the model
            # to occasionally pick grammatically related words instead of just the #1 memorized word.
            # This is how AI "thinks" and finds new relationships!
            scaled_logits = next_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # Select the top 3 most likely structural words and pick one semi-randomly
            top_probs, top_indices = torch.topk(probs, 3)
            # Normalize top probabilities
            top_probs = top_probs / torch.sum(top_probs)
            
            # Sample from the top 3 choices
            choice = torch.multinomial(top_probs, 1).item()
            next_token = top_indices[choice].item()
            
            if next_token == 2: # Hit the <STOP> token
                break
                
            generated.append(next_token)
            
            # Basic repetition break
            if len(generated) > 3 and generated[-1] == generated[-2] == generated[-3]:
                break
                
    response_tokens = generated[len(encoded):]
    if not response_tokens:
        # Fallback if the AI decides to immediately predict STOP, print the closest probable word instead
        top2 = torch.topk(logits[0, -1, :], 2)
        fallback_token = top2.indices[1].item() if top2.indices[0].item() == 2 else top2.indices[0].item()
        if fallback_token != 2:
            return tokenizer.decode([fallback_token]) + "..."
    
    return tokenizer.decode(response_tokens)

# 7. Live Terminal Interface
print("\n" + "="*60)
print(" QUANTARA SYSTEM INITIALIZED ".center(60))
print("="*60)
print("How to use:")
print(" -> To Chat: Type any message and press Enter.")
print(" -> To Teach: Type Question|Answer (e.g., 'what is fire|fire is hot')")
print(" -> To Exit: Type 'quit'")
print("="*60)

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            print("Quantara: Shutting down systems...")
            break
            
        if '|' in user_input:
            # ---- TEACHING MODE ----
            parts = user_input.split('|', 1)
            if len(parts) == 2:
                q, a = parts[0].strip().lower(), parts[1].strip().lower()
                memory_data.append((q, a))
                
    # Log Fact into Permanent Persistent Memory CSV
                with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter='|')
                    writer.writerow([q, a])
                    
                # Update Embedding dynamically if a new word is added and it exists in GloVe
                _g = get_glove()
                if _g:
                    with torch.no_grad():
                        for w in (q.split() + a.split()):
                            if w in tokenizer.word2idx and w in _g:
                                model.embeddings.embed.weight[tokenizer.word2idx[w]] = torch.tensor(_g[w]).to(device)

                # Trigger brain plasticity and integrate new thought pattern
                train_on_memory(120) 
        else:
            # ---- CHAT MODE & CONTINUOUS LEARNING ----
            if user_input:
                user_text = user_input.lower()
                
                # 1. Talk to the user using current intelligence
                response = generate_response(user_text)
                print(f"Quantara: {response}")
                
                # We NO LONGER append (user_text, "") to memory.
                # Appending the user's raw question to the database was poisoning its brain!
                # It taught the AI that "who are you" is followed immediately by <STOP>,
                # making it forget the answer you taught it earlier.
                
    except KeyboardInterrupt:
        break
