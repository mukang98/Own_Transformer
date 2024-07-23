import torch
from torch import nn

class Positional_Encoding(nn.Module):
    def __init__(self, max_length, dim, device):
        self.embedding =  torch.zero(max_length, dim, device=device) #(max_length*dim)
        # self.embedding.
        pos = torch.arange(0, max_length, device=device)
        pos = pos.float().unsqueeze(dim=1) # (max_length * 1)

        _2i = torch.arange(0, dim, step=2, device=device) # (dim/2)

        self.embedding[:, 0::2] = torch.sin(pos / 10000**_2i/dim)
        self.embedding[:, 1::2] = torch.cos(pos / 10000**_2i/dim)

    def forward(self, input):
        _, max_length = input.size()
        return self.embedding[:max_length,:]    