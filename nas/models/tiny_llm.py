import torch
import torch.nn as nn
import math

class TinyLLM(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, ff_dim, vocab_size, quantization_bits=8):
        super().__init__()
        self.quantization_bits = quantization_bits
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Use a smaller fixed-size positional encoding
        self.pos_encoding = self._create_pos_encoding(hidden_size, max_len=32)
        
        # Use parameter sharing between layers to reduce model size
        self.shared_transformer = TransformerLayer(hidden_size, num_heads, ff_dim)
        self.num_layers = num_layers
        
        # Reduce output layer size
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embedding + positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_len, :]
        
        # Reuse the same transformer layer multiple times
        for _ in range(self.num_layers):
            x = self.shared_transformer(x)
            
        return self.output(x)
    
    def _create_pos_encoding(self, hidden_size, max_len=512):
        pos_enc = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * 
                           -(math.log(10000.0) / hidden_size))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Self attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

