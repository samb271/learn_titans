import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.memory import MemoryModule
import numpy as np     

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float32) * 
                             (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, scale, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class MemoryAugmentedAttention(nn.Module):
    def __init__(self, dim_model, num_heads, dropout=0.1, memory_depth=2):
        super().__init__()
        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        
        self.d_k = dim_model // num_heads
        self.num_heads = num_heads
        
        # Linear projections
        self.Wq = nn.Linear(dim_model, dim_model)
        self.Wk = nn.Linear(dim_model, dim_model)
        self.Wv = nn.Linear(dim_model, dim_model)
        
        # Output projection
        self.linear_out = nn.Linear(dim_model, dim_model)
        
        # Attention mechanism
        self.attention = SelfAttention(dropout)
        
        # Memory module
        self.memory = MemoryModule(dim=dim_model, depth=memory_depth, heads=num_heads)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Retrieve memories using the input as query
        # Output memory tokens will be of identical shape to the input
        memories = self.memory.retrieve_memories(x)
        
        # Concatenate memory tokens with input tokens for enhanced context
        # This doubles the context window
        memory_enhanced_x = torch.cat([memories, x], dim=0)
        
        # Project to Q, K, V
        Q = self.Wq(memory_enhanced_x)
        K = self.Wk(memory_enhanced_x)
        V = self.Wv(memory_enhanced_x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.d_k)
        context, attn_weights = self.attention(Q, K, V, scale, mask)
        
        # Reshape output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.linear_out(context)
        
        # Extract only the part corresponding to the original sequence
        # Otherwise we would be using the predictions as targets in the loss function
        seq_len = x.shape[1]
        output_original = output[:, -seq_len:, :]
        
        # Memorize the output tokens (non-memory ones)
        # This is the update step in Fig. 2
        self.memory.memorize(output_original)
        
        return output_original, attn_weights

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, memory_depth=2, dropout=0.1):
        super().__init__()
        
        self.memory_attn = MemoryAugmentedAttention(
            d_model, num_heads, dropout, memory_depth)
        
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with memory
        residual = x
        x = self.norm1(x)
        x, _ = self.memory_attn(x, mask)
        x = self.dropout1(x)
        x = x + residual
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + residual
        
        return x

class Titans(nn.Module):
    def __init__(self, 
            vocab_size, 
            d_model, 
            num_heads, 
            num_layers, 
            d_ff, 
            max_seq_len, 
            memory_depth, 
            dropout,
            device
        ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, memory_depth, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        
        seq_len = x.size(1)
        
        # Create special mask which allows any memory token to attend to all other memory tokens
        # But regular tokens have regular causal mask
        mask = self.generate_causal_mask(seq_len, seq_len).to(self.device)
        
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        x = self.output_projection(x)
        
        return x
    
    def generate_causal_mask(self, m: int, c: int, device=None) -> torch.Tensor:
        
        N = m + c
        mask = torch.zeros((N, N), dtype=torch.bool, device=device)

        # Memory tokens (rows 0 to m-1): full access to memory tokens
        mask[:m, :m] = True

        # Causal tokens (rows m to N-1): access to memory + causal history
        mask[m:, :m] = True
        mask[m:, m:] = torch.tril(torch.ones((c, c), dtype=torch.bool, device=device))

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        generated = input_ids
        max_seq_len = self.positional_encoding.pe.size(1)
        
        for _ in range(max_new_tokens):
            # Truncate if needed
            if generated.size(1) >= max_seq_len:
                generated = generated[:, -(max_seq_len - 1):]
                
            # Get model predictions
            outputs = self.forward(generated)
            
            # Get next token logits
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k sampling if specified
            if top_k is not None:
                values, indices = torch.topk(next_token_logits, top_k)
                mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                mask.scatter_(1, indices, True)
                next_token_logits.masked_fill_(~mask, -float('inf'))
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
        return generated
