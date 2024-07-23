'''
1.文件名字改为更准确和优雅的
'''
import torch
from torch import nn
from token_embedding import TokenEmbedding
from postional_encoding import PositionalEncoding

# class Embedding(nn.Module):
#1. 起名不准确
class TransformerEmbedding(nn.Module):
    # def __init__(self):
    #     self.embedding = Input_Embedding()
    #     self.postion = Positional_Encoding()
    #     self.drouput = torch.dropout
    # 1.没有传参的意识，传参其实是自己对于每个part更准确的认识
    # 2.各种命名的问题
    # 3.忘记调用nn.Module的构造函数
    def __init__(self, vocab_size, dim, max_length, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, dim)
        self.pos_emb = PositionalEncoding(max_length, dim, device)
        self.drouput = nn.Dropout(p=drop_prob)
    # def forward(self, input):
    #     emb = self.embedding(input)
    #     postion =self.postion(input)
    #     dropout = self.drouput(0.9)
    #     return dropout(emb+postion)
    # 1.forward基本还可以
    def forward(self, input):
        tok_emb = self.tok_emb(input)
        pos_emb = self.pos_emb(input)
        return self.drouput(tok_emb + pos_emb)

#补充测试代码
#通过真正的code+review 发现了自己对于一些概念的模糊、一些命名的问题、python及少量torch语法的使用
# seq_length代表真实句子的长度，max_length代表模型可以处理的最大句子长度， vocab_size代表词汇表大小
if __name__ == "__main__":
    vocab_size = 10000
    embedding_dim = 512 
    input_tensor = torch.tensor([[1,4,226,844], [552,8855,4353,1]])
    embedding_layer = TransformerEmbedding(vocab_size, embedding_dim, max_length=256, drop_prob=0.9,device="cpu")
    output_tensor = embedding_layer(input_tensor)
    print("Input Tensor:")
    print(input_tensor)
    print("Output Tensor:")
    print(output_tensor)