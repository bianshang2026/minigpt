import torch
import torch.nn as nn
import math

class Selfattation(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
       
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads 

        assert self.head_dim * num_heads == output_dim, "output_dim must be divisible by num_heads"
        # Initialize weights and biases for the attention mechanism
        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        self.W_o = nn.Linear(output_dim, output_dim)


    def forward(self, x):
        # Compute queries, keys, and values
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        K = K.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        V = V.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        T = Q.shape[-2]  # Sequence length
        T1 = K.shape[-2]  # Sequence length
        mask = torch.tril(torch.ones(T, T1, device=x.device, dtype=torch.bool))  # Example mask for causal attention
        scores = scores.masked_fill(~mask, float('-inf'))  # Apply the mask to the scores
        attention_weights = torch.softmax(scores, dim=-1)

        # Compute the weighted sum of values
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(x.shape[0], x.shape[1], self.num_heads * self.head_dim)  # Reshape to (batch_size, output_dim)
        output = self.W_o(output) # Apply the output projection
        return output


