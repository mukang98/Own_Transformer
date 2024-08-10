import torch.nn as nn 

class PostionWiseFeedForward(nn.Module):
    """
    Output=Linear2(Dropout(ReLU(Linear1(x))))
    """
    def __init__(self, d_model, hidden_size, prob_drop = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(p=prob_drop)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        # out = self.linear2(out)
        # out = self.dropout(out) #1. Dropout应该放在激活函数之后，因为比如Relu函数之后数据会变得稀疏，因为负值被变为零。
        return out


