#For better understanding of mask in Transformer
#%%
# ================================================================= #
#                        1.src.mask                                 #
# ================================================================= #
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
# score = score.masked_fill(mask == 0, -10000) 

# %%
