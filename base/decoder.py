import torch
import torch.nn as nn
import math
from multihead_attention import MultiHeadAttention
from ffn import FeedForward
from encoder import EncoderLayer

# 解码器的代码实现
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 掩码的自注意力层
        self_attention_output = self.masked_self_attention(x, x, x, tgt_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = x + self_attention_output
        x = self.norm1(x)

        # 编码器-解码器注意力层
        enc_dec_attention_output = self.enc_dec_attention(x, encoder_output, encoder_output, src_mask)
        enc_dec_attention_output = self.dropout(enc_dec_attention_output)
        x = x + enc_dec_attention_output
        x = self.norm2(x)

        # 前馈层
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.norm3(x)

        return x


# 定义DecoderLayer的参数
d_model = 512  # 模型的维度
num_heads = 8  # 注意力头的数量
d_ff = 2048  # 前馈网络的维度
dropout = 0.1  # 丢弃概率
batch_size = 1  # 批量大小
max_len = 100  # 序列的最大长度



input_sequence = torch.randn(1, max_len, d_model)
encoder_layer = EncoderLayer(d_model, num_heads, d_ff, 0.1)
encoder_output = encoder_layer(input_sequence, None)


# 定义DecoderLayer实例
decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

src_mask = torch.rand(batch_size, max_len, max_len) > 0.5
tgt_mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0) == 0



# 将输入张量传递到DecoderLayer
output = decoder_layer(input_sequence, encoder_output, src_mask, tgt_mask)

# 输出形状
print("Output shape:", output.shape)