import torch
import torch.nn as nn
import math

class EmbeddingMap(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        # output shape: [batch_size, seq_len, embedding_dim]
        # Scale embeddings by sqrt(d_model) as per Attention is All You Need
        embeddings = self.embed(x) * math.sqrt(self.embedding_dim)
        
        # Add Positional Encoding
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=x.device, dtype=torch.float) * (-math.log(10000.0) / self.embedding_dim))
        
        pe = torch.zeros(seq_len, self.embedding_dim, device=x.device)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        
        return embeddings + pe.unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Split into multiple heads: [batch_size, num_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention across 4 different "thought streams" simultaneously
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # AUTOREGRESSIVE MASK (The "No Cheating" rule)
        # Prevents the model from looking at future words when predicting the next word
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Combine the 4 heads back together into the original dimension
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out_proj(output)

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        hidden_dim = embedding_dim * 4
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x):
        # Residual connection included at block level
        return self.linear2(self.relu(self.linear1(x)))

class PerspectiveNode(nn.Module):
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = FeedForward(embedding_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        # Attention with residual and norm
        attn_out = self.attention(self.layer_norm1(x))
        x = x + attn_out
        
        # FFN with residual and norm
        ffn_out = self.ffn(self.layer_norm2(x))
        x = x + ffn_out
        return x

class OutputLayer(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        return self.linear(x)

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_layers=1):
        super().__init__()
        self.embeddings = EmbeddingMap(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([PerspectiveNode(embedding_dim) for _ in range(num_layers)])
        self.output_layer = OutputLayer(embedding_dim, vocab_size)
        
    def forward(self, x):
        h = self.embeddings(x)
        for layer in self.layers:
            h = layer(h)
        logits = self.output_layer(h)
        return logits
