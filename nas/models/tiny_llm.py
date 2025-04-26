import torch
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TinyLLM(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, ff_dim, vocab_size, quantization_bits=8):
        super().__init__()
        # Ensure hidden_size is divisible by num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.seq_length = 32  # Fixed sequence length
        self.quantization_bits = quantization_bits
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = self._create_pos_encoding(hidden_size, max_len=self.seq_length)
        
        # parameter sharing between layers reduces model size
        self.shared_transformer = TransformerLayer(hidden_size, num_heads, ff_dim)
        self.num_layers = num_layers
        
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = x.to(device)
        # Ensure input sequence length matches model's sequence length
        if x.size(1) != self.seq_length:
            x = self._pad_or_truncate(x)
            
        # embedding + positional encoding
        x = self.embedding(x)
        x = x + self.pos_encoding.to(device)
        
        for _ in range(self.num_layers):
            x = self.shared_transformer(x)
            
        return self.output(x)
    
    def _pad_or_truncate(self, x):
        """Pad or truncate input sequence to match model's sequence length"""
        batch_size = x.size(0)
        if x.size(1) < self.seq_length:
            # Pad
            padding = torch.zeros((batch_size, self.seq_length - x.size(1)), 
                                dtype=torch.long, device=x.device)
            x = torch.cat([x, padding], dim=1)
        else:
            # Truncate
            x = x[:, :self.seq_length]
        return x
    
    def _create_pos_encoding(self, hidden_size, max_len=32):
        pos_enc = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
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
        # Reshape for attention layer
        x_t = x.transpose(0, 1)  # [seq_len, batch, hidden_size]
        attn_out, _ = self.attention(x_t, x_t, x_t)
        attn_out = attn_out.transpose(0, 1)  # [batch, seq_len, hidden_size]
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x
