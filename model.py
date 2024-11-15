import torch
import math

class InputEmbeddings(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """ 
        d_model: dimension of vector
        vocab_size: # words in vocab
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = torch.nn.Dropout(dropout)

        # positional encoding:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # sin to even positions:
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # add a batch dimension (1, seq_len, d_model)

        self.register_buffer('pe', pe) # will save tensor 

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # this is always the same, so no gradients
        return self.dropout(x)


class LayerNormalization(torch.nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = torch.nn.Parameter(torch.ones(1)) # makes it learnable
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))