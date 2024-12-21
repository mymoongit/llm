'''
前馈神经网络
'''
import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention

# 前馈网络的代码实现
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 线性变换1
        x = self.relu(self.linear1(x))

        # 线性变换2
        x = self.linear2(x)

        return x


# 示例用法
d_model = 512
max_len = 100
num_heads = 8
d_ff = 4 * d_model

# 多头注意力
multihead_attn = MultiHeadAttention(d_model, num_heads)

# 前馈网络
ff_network = FeedForward(d_model, d_ff)

# 示例输入序列
input_sequence = torch.randn(5, max_len, d_model)

# 多头注意力
attention_output = multihead_attn(input_sequence, input_sequence, input_sequence)

# 前馈网络
output_ff = ff_network(attention_output)
print('input_sequence', input_sequence.shape)
print("output_ff", output_ff.shape)