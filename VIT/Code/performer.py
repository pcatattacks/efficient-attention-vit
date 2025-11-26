import torch
import torch.nn as nn
from performer_pytorch import FastAttention
import math

class PerformerAttention(nn.Module):
    """Performer attention using FAVOR+ mechanism
    
    Implementation adapted from 
    https://github.com/mlpen/Nystromformer/blob/main/reorganized_code/encoders/backbones/efficient_attentions/attention_performer.py
    """

    def __init__(self, config):
        super().__init__()

        # linformer-like config
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # head_dim is same as attention_head_size in vit.py
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rp_dim = config["rp_dim"]
        self.kernel_type = config["kernel_type"]
        if self.kernel_type == "relu":
            self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, kernel_fn = nn.ReLU())
        elif self.kernel_type == "exp":
            self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, kernel_fn = torch.exp)

        # Reuse qkv projection code from standard attention
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        all_head_size = self.num_attention_heads * self.head_dim
        self.qkv_projection = nn.Linear(self.hidden_size, all_head_size * 3, bias=self.qkv_bias)

    def forward(self, x, output_attentions=False):
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        Q = query.view(batch_size, sequence_length, self.num_attention_heads,   self.head_dim).transpose(1, 2)
        K = key.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        V = value.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        attn_output = self.attn_fn(
            Q / math.sqrt(math.sqrt(self.head_dim)),
            K / math.sqrt(math.sqrt(self.head_dim)),
            V)
        
        # Reshape: (batch, heads, seq, head_dim) -> (batch, seq, hidden_size)
        batch_size, _, sequence_length, _ = attn_output.shape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, self.hidden_size)
        
        return attn_output, None  # No attention weights returned

    def extra_repr(self):
        return f'rp_dim={self.rp_dim}, kernel_type={self.kernel_type}'