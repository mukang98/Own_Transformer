import torch.nn as nn

from models.layers.multihead_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PostionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_size, head_nums, prob_drop) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, head_nums)
        self.norm = LayerNorm(d_model)
        self.fc = PostionWiseFeedForward(d_model, hidden_size)
        self.dropout = nn.Dropout(p=prob_drop)

    def forward(self, x, mask):
        _x = x
        out = self.attention(q = x, k = x, v = x, mask=mask) #Multihead-attention

        out = self.dropout(out)
        out = self.norm(out + _x) #add+norm 
        _out = out 

        out = self.fc(out)
        out = self.dropout(out)
        out += _out
        return out


