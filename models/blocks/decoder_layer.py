import torch.nn as nn

from models.layers.multihead_attention import MultiHeadAttention
from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PostionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, head_nums,prob_drop, hidden_size):
        super().__init__()
        self.dec_attention = MultiHeadAttention(d_model, head_nums)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=prob_drop)
        
        self.enc_dec_attention = MultiHeadAttention(d_model, head_nums)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=prob_drop)

        self.ffn = PostionWiseFeedForward(d_model, hidden_size,prob_drop)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=prob_drop)
    
    def forward(self, enc, dec, trg_mask, src_mask):
        _out = dec
        out = self.dec_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        out = self.dropout1(out)
        out = self.norm1(out + _out)

        if enc is not None: #1. 没考虑为None的情况
            _out = out
            out = self.enc_dec_attention(q=dec, k=enc, v=enc, mask = src_mask) #1. 忘记这里的mask
            out = self.dropout2(out)
            out = self.norm2(out + _out)

        _out = out 
        out = self.ffn(out)
        out = self.dropout3(out)
        out = self.norm3(out + _out)

        return out


