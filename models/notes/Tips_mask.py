#For better understanding of mask in Transformer
#%%
# ================================================================= #
#                        1.src.mask                                 #
# ================================================================= #
#src_mask prevents the model from attending to certain positions in the source sequence,
sentence = "Hello world"
import torch
import torch.nn as nn
vocab_size = 3
# Assuming that the vacab_size is like 
#<pad>
#Hello
#world
embedding_dim = 1
embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
#padding mask is normally used to identify the index of the padding token.
input_dices = torch.tensor([1, 2, 0, 0]) 
output_embedding = embedding(input_dices)
print(output_embedding) 
print(output_embedding.shape)#torch.Size([4, 1])
q = k = output_embedding
qk = torch.matmul(q,k.t())
print(qk)
src_mask = torch.tensor([1,1,0,0])
qk_masked = qk.masked_fill(src_mask == 0, -1e-9) 
print(qk_masked)
# %%
# ================================================================= #
#                        1.trg.mask                                 #
# ================================================================= #
# trg_mask prevents the model from attending to future positions in the target sequence during training.
import torch
import numpy as np
trg_mask = torch.from_numpy(np.array([[ True, False, False, False],
                                      [ True,  True, False, False],
                                      [ True,  True, True, False],
                                      [ True,  True,  True, True]])).float() * 1
print(trg_mask)
qk_masked = qk.masked_fill(trg_mask== 0, -1e-9) 
print(qk_masked)
# %%
