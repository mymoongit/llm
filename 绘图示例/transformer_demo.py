import torch
import torch.nn as nn
import torch.optim as optim

# 定义自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        N, seq_length, d_model = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.reshape(N, seq_length, self.num_heads, self.d_k)
        K = K.reshape(N, seq_length, self.num_heads, self.d_k)
        V = V.reshape(N, seq_length, self.num_heads, self.d_k)

        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        attention = torch.softmax(energy / (self.d_k ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, V])
        out = out.reshape(N, seq_length, d_model)

        return self.fc_out(out)

# 定义 Transformer 编码器层
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(attn_output + x)
        ff_output = self.ff(x)
        x = self.norm2(ff_output + x)
        return self.dropout(x)

# 定义 Transformer 编码器
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        # 列表生成式
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)])

        self.embed = nn.Linear(input_dim, d_model)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x

# 示例数据集
data = torch.rand(10, 5, 8)  # (batch_size, seq_length, input_dim)

# 模型实例
'''
d_model 是每个token特征向量维度，如 128、256、512、768 或 1024
每一个block输出都是同一个d_model
自注意力模块会将 512 维的输入分成 8 个头，每个头的维度是 64
所有的中间结果（自注意力输出、残差连接、FFN 输出）都保持 512 维
'''


model = TransformerEncoder(input_dim=8, d_model=32, num_layers=1, num_heads=4, dropout=0.1)

# 前向传播
output = model(data)


# print(model)
print(output.shape)
# print(data)