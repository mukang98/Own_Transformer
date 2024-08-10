import torch.nn as nn

from models.embedding.transformer_embedding import TransformerEmbedding
from models.blocks.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_length, drop_prob, device, hidden_size, head_nums, prob_drop, n_layers):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, 
                                        d_model, 
                                        max_length, 
                                        drop_prob, 
                                        device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, 
                                                  hidden_size, 
                                                  head_nums, 
                                                  prob_drop)] 
                                    for _ in range(n_layers))

    def forward(self, x, src_mask):
        out = self.emb(x)
        for layer in self.layers:
            out = layer(out, src_mask)
        return out
