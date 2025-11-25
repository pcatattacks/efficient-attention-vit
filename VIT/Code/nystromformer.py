import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NystromformerAttention(nn.Module):
    """Nyström method for attention matrix approximation"""
    # Approximates attention matrix using landmark points
    # Balances efficiency and approximation quality
    # Implementation modified from https://github.com/mlpen/Nystromformer (original paper implementation).


    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        # self.head_dim = config["head_dim"]

        # same as attention_head_size in vit.py
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.num_landmarks = config["num_landmarks"]
        # self.seq_len = config["max_seq_len"]

        self.use_conv = "conv_kernel_size" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_attention_heads, out_channels = self.num_attention_heads,
                kernel_size = (config["conv_kernel_size"], 1), padding = (config["conv_kernel_size"] // 2, 0),
                bias = False,
                groups = self.num_attention_heads)
        
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
        

    def forward(self, x, output_attentions=False):

        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        Q = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        K = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        V = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        if self.num_landmarks == sequence_length:
            attn = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.attention_head_size), dim = -1)
            X = torch.matmul(attn, V)
        else:
            # Pad sequence to be evenly divisible by num_landmarks
            seqlen = sequence_length
            if seqlen % self.num_landmarks != 0:
                padding_len = self.num_landmarks - (seqlen % self.num_landmarks)
                # Pad along the sequence dimension
                Q = F.pad(Q, (0, 0, 0, padding_len))
                K = F.pad(K, (0, 0, 0, padding_len))
                V = F.pad(V, (0, 0, 0, padding_len))
                sequence_length = seqlen + padding_len
            
            Q_landmarks = Q.reshape(-1, self.num_attention_heads, self.num_landmarks, sequence_length // self.num_landmarks, self.attention_head_size).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_attention_heads, self.num_landmarks, sequence_length // self.num_landmarks, self.attention_head_size).mean(dim = -2)

            kernel_1 = F.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)) / math.sqrt(self.attention_head_size), dim = -1)
            kernel_2 = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)) / math.sqrt(self.attention_head_size), dim = -1)
            kernel_3 = F.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) / math.sqrt(self.attention_head_size), dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

            # Remove padding from output
            if seqlen % self.num_landmarks != 0:
                X = X[:, :, :seqlen, :]
                V = V[:, :, :seqlen, :]
                sequence_length = seqlen # go back to the non-padded sequence length

        if self.use_conv:
            X += self.conv(V)

        # Reshape from (batch, num_attention_heads, seq_len, head_dim) to (batch, seq_len, hidden_size)
        X = X.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.all_head_size)
        X = self.output_projection(X)
        X = self.output_dropout(X)

        if not output_attentions:
            return (X, None)
        else:
            # note this doesn't return attention maps when num_landmarks < seq_len
            # nystrom attention maps are not well-defined in that case
            return (X, attn if self.num_landmarks == sequence_length else None)
        

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        V = 1 / (torch.max(torch.sum(torch.abs(K), dim = -2)) * torch.max(torch.sum(torch.abs(K), dim = -1))) * K.transpose(-1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V