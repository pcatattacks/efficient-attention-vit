import math
import torch
from torch import nn

class LinformerAttentionHead(nn.Module):
    """
    A single Linformer-style self-attention head.

    Args:
        hidden_size: input dim (per token), usually d_model.
        attention_head_size: per-head dim d_head.
        k: projected sequence length (L -> k).
        seq_len: maximum sequence length L (including CLS token).
    """
    def __init__(self, hidden_size, attention_head_size, k, seq_len, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.k = k
        self.seq_len = seq_len

        # Standard Q, K, V projections (same as full attention)
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key   = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        # Linformer sequence projections: (k, L)
        # These compress along the *sequence* dimension.
        self.E_K = nn.Parameter(
            torch.randn(k, seq_len) * (1.0 / seq_len ** 0.5)
        )
        self.E_V = nn.Parameter(
            torch.randn(k, seq_len) * (1.0 / seq_len ** 0.5)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_size)
        returns:
            attention_output: (batch_size, seq_len, attention_head_size)
            attention_probs: (batch_size, seq_len, k)
        """
        batch_size, L, _ = x.shape
        if L > self.seq_len:
            raise ValueError(f"Input sequence length {L} exceeds configured seq_len={self.seq_len}")
        # In case L < seq_len (e.g. future generalization), slice the projections
        E_K = self.E_K[:, :L]  # (k, L)
        E_V = self.E_V[:, :L]  # (k, L)

        # 1) Q, K, V: (B, L, hidden_size) -> (B, L, d_head)
        query = self.query(x)
        key   = self.key(x)
        value = self.value(x)

        # 2) Project K, V along sequence: (B, L, d_head) -> (B, k, d_head)
        # einsum: (k, L) x (B, L, d) -> (B, k, d)
        K_proj = torch.einsum("kl,bld->bkd", E_K, key)
        V_proj = torch.einsum("kl,bld->bkd", E_V, value)

        # 3) Attention over compressed K: (B, L, d) x (B, k, d)^T -> (B, L, k)
        attention_scores = torch.matmul(
            query, K_proj.transpose(-1, -2)
        ) / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 4) Weighted sum over compressed V: (B, L, k) x (B, k, d) -> (B, L, d)
        attention_output = torch.matmul(attention_probs, V_proj)

        return attention_output, attention_probs


class LinformerMultiHeadAttention(nn.Module):
    """
    Multi-head Linformer attention.
    Each head is a LinformerAttentionHead.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]

        # Linformer-specific config
        self.k = config["linformer_k"]          # projected sequence length
        self.seq_len = config["seq_len"]        # max sequence length (CLS + patches)

        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = LinformerAttentionHead(
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_head_size,
                k=self.k,
                seq_len=self.seq_len,
                dropout=config["attention_probs_dropout_prob"],
                bias=self.qkv_bias,
            )
            self.heads.append(head)

        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # x: (B, L, hidden_size)
        attention_outputs = [head(x) for head in self.heads]
        # concat over head dim
        attention_output = torch.cat(
            [att_out for att_out, _ in attention_outputs],
            dim=-1
        )  # (B, L, all_head_size)

        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return attention_output, None
        else:
            # stack head-wise probs: (B, num_heads, L, k)
            attention_probs = torch.stack(
                [att_probs for _, att_probs in attention_outputs],
                dim=1,
            )
            return attention_output, attention_probs

class LinformerFasterMultiHeadAttention(nn.Module):
    """
    Multi-head Linformer attention with fused QKV projection.

    This is the Linformer analogue of FasterMultiHeadAttention:
    - One big linear for QKV
    - Shared sequence projections E_K, E_V across heads
    - Complexity: O(B * h * L * k * d_head) instead of O(B * h * L^2 * d_head)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]

        # Linformer-specific
        self.k = config["linformer_k"]   # projected sequence length
        self.seq_len = config["seq_len"] # max sequence length (CLS + patches)

        # Fused QKV projection
        self.qkv_projection = nn.Linear(
            self.hidden_size,
            self.all_head_size * 3,
            bias=self.qkv_bias,
        )

        # Shared sequence projections (k x L)
        self.E_K = nn.Parameter(
            torch.randn(self.k, self.seq_len) * (1.0 / self.seq_len ** 0.5)
        )
        self.E_V = nn.Parameter(
            torch.randn(self.k, self.seq_len) * (1.0 / self.seq_len ** 0.5)
        )

        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        """
        x: (batch_size, seq_len, hidden_size)
        returns:
            attention_output: (batch_size, seq_len, hidden_size)
            attention_probs:  (batch_size, num_heads, seq_len, k) or None
        """
        # 1) Fused QKV
        # (B, L, H) -> (B, L, 3 * H)
        qkv = self.qkv_projection(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        batch_size, L, _ = query.size()
        if L > self.seq_len:
            raise ValueError(
                f"Input sequence length {L} exceeds configured seq_len={self.seq_len}"
            )

        # (B, L, H) -> (B, h, L, d_head)
        query = query.view(batch_size, L, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key   = key.view(batch_size, L, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, L, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # 2) Linformer projection along sequence for K and V
        # Slice E_K/E_V in case L < seq_len
        E_K = self.E_K[:, :L]  # (k, L)
        E_V = self.E_V[:, :L]  # (k, L)

        # Flatten heads to apply einsum once
        # key_flat:   (B*h, L, d_head)
        # value_flat: (B*h, L, d_head)
        key_flat = key.reshape(batch_size * self.num_attention_heads, L, self.attention_head_size)
        value_flat = value.reshape(batch_size * self.num_attention_heads, L, self.attention_head_size)

        # K_proj_flat: (B*h, k, d_head)
        # einsum: (k, L) x (B*h, L, d) -> (B*h, k, d)
        K_proj_flat = torch.einsum("kl,bld->bkd", E_K, key_flat)
        V_proj_flat = torch.einsum("kl,bld->bkd", E_V, value_flat)

        # Reshape back to (B, h, k, d_head)
        K_proj = K_proj_flat.view(batch_size, self.num_attention_heads, self.k, self.attention_head_size)
        V_proj = V_proj_flat.view(batch_size, self.num_attention_heads, self.k, self.attention_head_size)

        # 3) Attention over compressed K
        # query:   (B, h, L, d)
        # K_proj:  (B, h, k, d)
        # scores:  (B, h, L, k)
        attention_scores = torch.matmul(
            query,
            K_proj.transpose(-1, -2)  # (B, h, d, k)
        ) / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        # 4) Weighted sum over compressed V
        # (B, h, L, k) x (B, h, k, d) -> (B, h, L, d)
        attention_output = torch.matmul(attention_probs, V_proj)

        # 5) Merge heads and project back
        # (B, h, L, d) -> (B, L, h*d)
        attention_output = (
            attention_output
            .transpose(1, 2)              # (B, L, h, d)
            .contiguous()
            .view(batch_size, L, self.all_head_size)
        )

        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return attention_output, None
        else:
            # (B, h, L, k)
            return attention_output, attention_probs