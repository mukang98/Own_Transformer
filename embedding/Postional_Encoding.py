import torch
from torch import nn

# class Positional_Encoding(nn.Module): 1.类名不用下划线 2.此外也将文件名都改为了小写
class PositionalEncoding(nn.Module):
    def __init__(self, max_length, dim, device):

        super().__init__() # 1.忘记调用nn.Module的构造函数


        # self.embedding =  torch.zero(max_length, dim, device=device) #(max_length*dim)
        #  1.写错了调用的函数 2.起名歧义
        self.encoding =  torch.zeros(max_length, dim, device=device) #(max_length*dim)
        # self.embedding.
        # 1.禁用梯度的方法忘记了
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_length, device=device)
        pos = pos.float().unsqueeze(dim=1) # (max_length * 1)

        # _2i = torch.arange(0, dim, step=2, device=device) # (dim/2) 
        # 1.忘记统一为浮点数了 
        _2i = torch.arange(0, dim, step=2, device=device).float() # (dim/2)
        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / dim))
        self.encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / dim))

    def forward(self, input):
        # _, max_length = input.size() # 1.起名不能表达真实的含义
        _, seq_length = input.size()
        return self.encoding[:seq_length,:]    