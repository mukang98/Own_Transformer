"""
为了更好地理解transformer中的每一处细节,每一块涉及到的未能在代码中体现的知识点将体现于此。
包括特定的torch库、torch函数、公式等
"""
# ================================================================= #
#                        1.nn.Embedding                             #
# ================================================================= #
#%%
import torch
import torch.nn as nn
vocab_size = 10000
embedding_dim = 512
embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1) 
#nn.Embedding接收的第一个参数是词汇表的大小，第二个是嵌入向量的维度，padding_idx是为了统一输入长度的填充即对于index为1的输入置0
input_dices = torch.tensor([[1,4,226,844], [552,8855,4353,1]]) #B*N
#通过nn.Embedding创建的embedding方法，接收的输入是每行代表每句话，每列代表每句话的每个token 
output_embedding = embedding(input_dices) #B*N*S
print(output_embedding) 
#通过nn.Embedding创建的embedding方法,输出是对于每一个token代表的index扩充成嵌入向量的维度。
# 输出的每一块代表每句话，每块的每一行代表每句话的每个token，每一列代表嵌入向量的维度一个维度的值。
print(output_embedding.shape)#torch.Size([2, 4, 512])
# %%
pos = torch.arange(0, 6)
print(pos)
pos = pos.float().unsqueeze(dim=1)
print(pos.shape)
# %%
max_len = 4
d_model = 12
encoding = torch.zeros(max_len, d_model)
pos = torch.arange(0, max_len)
pos = pos.float().unsqueeze(dim=1)
print(pos)
_2i = torch.arange(0, d_model, step=2)
print(f'_2i是:{_2i}')
encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
print(pos+_2i)
# print(_2i)
# print(pos)
# print(encoding)
# encoding[:, 0::2] = pos
# print(encoding)

# %%
