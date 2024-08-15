from torch import nn
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    # def forward(self, q, k, v, mask): 1.mask默认为None
    def forward(self,q, k, v, mask=None):
        # batch_size, head_nums, seq_length, d_model = k.size() :1.命名问题 应该是d_tensor
        batch_size, head_nums, seq_length, d_tensor = k.size() #B*H*N*S
        
        k_t = k.transpose(2,3)
        # score = q @ k / math.sqrt(d_model)  1.缺少括号
        score (q @ k) / math.sqrt(d_tensor) 
 
        # if mask:  1. 与mask=None对应
        if mask is not None: 
            score = score.masked_fill(mask == 0, -10000) #float("-inf")
        score = self.softmax(score) #B*H*N*N
        v = score @ v #B*H*N*S
        return v, score 