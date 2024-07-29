#%%
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
