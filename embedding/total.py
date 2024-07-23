import torch
from torch import nn
from Embedding import Input_Embedding
from Postional_Encoding import Positional_Encoding

class Embedding(nn.Module):
    def __init__(self):
        self.embedding = Input_Embedding()
        self.postion = Positional_Encoding()
        self.drouput = torch.dropout
    def forward(self, input):
        emb = self.embedding(input)
        postion =self.postion(input)
        dropout = self.drouput(0.9)
        return dropout(emb+postion)