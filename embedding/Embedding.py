import torch.nn as nn

class Input_Embedding(nn.Embedding):
    def __init__(self, vocab_size, dim):
        super().__init__(vocab_size, dim, padding_idx=1)