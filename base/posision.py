'''
位置编码
'''
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # 存为常量，不会被许梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# 示例用法
d_model = 512
max_len = 100
num_heads = 8

# 位置编码
pos_encoder = PositionalEncoding(d_model, max_len)

# 示例输入序列
input_sequence = torch.randn(5, max_len, d_model)

# 应用位置编码
input_sequence = pos_encoder(input_sequence)
print("输入序列的位置编码:")
print(input_sequence.shape)
