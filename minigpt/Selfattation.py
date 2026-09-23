import torch
import torch.nn as nn

class Selfattation(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, batch_size=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads 
        self.batch_size = batch_size
        assert self.head_dim * num_heads == output_dim, "output_dim must be divisible by num_heads"
        # Initialize weights and biases for the attention mechanism
        self.W_q = nn.Parameter(torch.randn(input_dim, output_dim))
        self.W_k = nn.Parameter(torch.randn(input_dim, output_dim))
        self.W_v = nn.Parameter(torch.randn(input_dim, output_dim))
        self.W_o = nn.Parameter(torch.randn(output_dim, output_dim))

    def forward(self, x):
        # Compute queries, keys, and values
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)
        Q = Q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        K = K.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        V = V.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2]))  # Example mask for causal attention
        scores = scores.masked_fill(mask == 0, float('-inf'))  # Apply the mask to the scores
        attention_weights = self.softmax(scores)

        # Compute the weighted sum of values
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(x.shape[0], x.shape[1], self.num_heads * self.head_dim)  # Reshape to (batch_size, output_dim)
        output = torch.matmul(output, self.W_o)  # Apply the output projection
        return output

    def softmax(self, x):
        softmax = nn.Softmax(dim=-1)
        return softmax(x)
