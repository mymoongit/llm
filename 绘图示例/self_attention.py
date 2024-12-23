import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据
d_model = 8
seq_length = 5
num_heads = 2

# 模拟 Query, Key, Value
np.random.seed(42)
Q = np.random.rand(seq_length, d_model)
K = np.random.rand(seq_length, d_model)
V = np.random.rand(seq_length, d_model)

# 计算能量值
energy = np.dot(Q, K.T)

# 计算注意力权重
attention_weights = np.exp(energy) / np.sum(np.exp(energy), axis=1, keepdims=True)

# 绘制注意力权重图
plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights, annot=True, cmap="viridis", xticklabels=range(seq_length), yticklabels=range(seq_length))
plt.title("Self-Attention Weights")
plt.xlabel("Key Positions")
plt.ylabel("Query Positions")
plt.show()