import torch.nn as nn

# class Input_Embedding(nn.Embedding): 1.命名歧义 2.类名不用下划线
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, dim):
        super().__init__(vocab_size, dim, padding_idx=1)