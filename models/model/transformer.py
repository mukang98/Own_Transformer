import torch.nn as nn
import torch
from models.model.decoder import Decoder
from models.model.encoder import Encoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, vocab_size, d_model, max_length, drop_prob, device, hidden_size, head_nums, prob_drop, n_layers, decoder_size):
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.encoder = Encoder(vocab_size=vocab_size, 
                               d_model=d_model, 
                               max_length=max_length, 
                               drop_prob=drop_prob, 
                               device=device, 
                               hidden_size=hidden_size, 
                               head_nums=head_nums, 
                               prob_drop=prob_drop, 
                               n_layers=n_layers)
        self.decoder = Decoder(vocab_size=vocab_size, 
                               max_length=max_length, 
                               drop_prob=drop_prob, 
                               decoder_size=device,
                               d_model=d_model, 
                               head_nums=head_nums,
                               prob_drop=prob_drop, 
                               hidden_size=hidden_size, 
                               n_layers=n_layers, 
                               decoder_size=decoder_size)
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)    
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.bool).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask