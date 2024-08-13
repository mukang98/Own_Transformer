import torch.nn as nn

from models.embedding.transformer_embedding import TransformerEmbedding
from models.blocks.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, vocab_size, dim, max_length, drop_prob, device,d_model, head_nums,prob_drop, hidden_size, n_layers, decoder_size):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size, 
                                        dim, 
                                        max_length, 
                                        drop_prob, 
                                        device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, 
                                   head_nums,
                                   prob_drop, 
                                   hidden_size)
                        for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, decoder_size)
    def forward(self, dec, enc, trg_mask, src_mask):
        out = self.emb(dec)
        for layer in self.layers:
            out = layer(out, enc, trg_mask, src_mask)
            
        out = self.linear(out)

        return out
        