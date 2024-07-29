from torch import nn
from scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_nums): # 其实可以写成n_head与d_model 保持一致
        super().__init__()
        self.head = head_nums
        self.dot_product = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.concat_out = nn.Linear(d_model, d_model)
    # def forward(self, q, k, v): 1.缺少了mask
    def forward(self, q, k, v, mask=None):
        #为了书写的简洁
        # q = self.w_q(q)
        # k = self.w_k(k)
        # v = self.w_v(v)
        # q = self.split(q)
        # k = self.split(k)
        # v = self.split(v)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.dot_product(q, k ,v, mask=mask)
        concat = self.concat(out)
        concat_out = self.concat_out(concat)
        return concat_out

    def split(self, input):
        batch_size, seq_length , d_model = input.size()
        d_tensor = d_model // self.head
        input = input.view(batch_size, seq_length, self.head, d_tensor).transpose(1,2)
        return input

    def concat(self, input):
        batch_size, head_nums, seq_length, d_tensor = input.size()
        d_model = head_nums * d_tensor
        input = input.transpose(1,2).contiguous().view(batch_size, seq_length, d_model)
        return input
        