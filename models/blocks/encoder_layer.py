import torch.nn as nn

from models.layers.multihead_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PostionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_size, head_nums, prob_drop):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, head_nums=head_nums)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=prob_drop)

        self.fc = PostionWiseFeedForward(d_model=d_model, hidden_size=hidden_size)
        self.norm2 = LayerNorm(d_model=d_model) # LayerNorm 和 Dropout 是两次独立的操作，分别处理不同的部分
        self.dropout2 = nn.Dropout(p=prob_drop)

    def forward(self, x, mask):
        _x = x
        out = self.attention(q = x, k = x, v = x, mask=mask) #Multihead-attention

        out = self.dropout1(out)
        out = self.norm1(out + _x) #add+norm 
        _out = out 

        out = self.fc(out)
        out = self.dropout2(out)
        out = self.norm2(out + _out)
        # out += _out #1.缺了最后一个norm
        return out


