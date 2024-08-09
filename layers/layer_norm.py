import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-12) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, unbiased = False, keepdim = True)

        # out = (x - mean) / torch.sqrt(var + self.eps) * 1. self.gamma + self.beta #分两行写，逻辑清晰和美观
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

