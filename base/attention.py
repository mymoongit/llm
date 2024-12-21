import torch
import torch.nn.functional as F

input_sequence = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.4], [0.7, 0.8, 0.9, 0.4]])

# 生成 Key、Query 和 Value 矩阵的随机权重
random_weights_key = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_query = torch.randn(input_sequence.size(-1), input_sequence.size(-1))
random_weights_value = torch.randn(input_sequence.size(-1), input_sequence.size(-1))

# 计算 Key、Query 和 Value 矩阵
query = torch.matmul(input_sequence, random_weights_query)
key = torch.matmul(input_sequence, random_weights_key)

value = torch.matmul(input_sequence, random_weights_value)


# 计算注意力分数
attention_scores = torch.matmul(query, key.T) / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
print(query.shape, key.shape)
print(attention_scores.shape)

# 使用 softmax 函数获得注意力权重
attention_weights = F.softmax(attention_scores, dim=-1)

# 计算 Value 向量的加权和
output = torch.matmul(attention_weights, value)

print("自注意力机制后的输出:")
print(output)