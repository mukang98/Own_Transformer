#%%
# ================================================================= #
#                        1.x.masked_fill                            #
# ================================================================= #
import torch
# 创建一个示例张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 创建一个掩码张量
mask = torch.tensor([[0, 1, 0], [1, 0, 5]])
# 使用 masked_fill，将 mask 为 1 的位置填充为 -1
x_filled = x.masked_fill(mask == 1, -100)
#掩码张量与示例张量的size需要相同
print(x_filled)
# %%
# ================================================================= #
#                        2.nn.Parameter                             #
# ================================================================= #
import torch
import torch.nn as nn
import math

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        # 定义权重和偏置参数
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重和偏置
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # 实现线性变换
        return torch.matmul(input, self.weight.t()) + self.bias

# 使用示例
input_features = 5
output_features = 3
batch_size = 2

model = CustomLinear(input_features, output_features)
input_data = torch.randn(batch_size, input_features)
print(input_data)
# 前向传播
output = model(input_data)
print(output)

# 查看所有参数
for param in model.parameters():
    print(param)

# 查看参数名和参数
for name, param in model.named_parameters():
    print(f"Parameter name: {name}")
    print(param)

# %%
