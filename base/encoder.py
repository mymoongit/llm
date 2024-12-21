import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from ffn import FeedForward


# 编码器的代码实现
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力层
        attention_output= self.self_attention(x, x, x, mask)
        attention_output = self.dropout(attention_output)
        x = x + attention_output
        x = self.norm1(x)

        # 前馈层
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.norm2(x)

        return x

d_model = 512
max_len = 100
num_heads = 8
d_ff = 2048


# 多头注意力
encoder_layer = EncoderLayer(d_model, num_heads, d_ff, 0.1)

# 示例输入序列
input_sequence = torch.randn(1, max_len, d_model)

# 多头注意力
encoder_output= encoder_layer(input_sequence, None)
print("encoder output shape:", encoder_output.shape)