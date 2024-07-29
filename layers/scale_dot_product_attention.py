from torch import nn
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask):
        batch_size, head_nums, seq_length, dim_model = k.size()
        k_t = k.transpose(2,3)
        score = q @ k / math.sqrt(dim_model)

        if mask:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score