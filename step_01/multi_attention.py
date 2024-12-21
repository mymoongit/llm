import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据
d_model = 8
seq_length = 5
num_heads = 2

# 示例多头数据
num_heads = 4
d_k = d_model // num_heads

# 模拟多头 Query, Key, Value
Q_heads = np.random.rand(seq_length, num_heads, d_k)
K_heads = np.random.rand(seq_length, num_heads, d_k)
V_heads = np.random.rand(seq_length, num_heads, d_k)

attention_heads = []
for i in range(num_heads):
    energy_head = np.dot(Q_heads[:, i, :], K_heads[:, i, :].T)
    attention_head = np.exp(energy_head) / np.sum(np.exp(energy_head), axis=1, keepdims=True)
    attention_heads.append(attention_head)

# 绘制每个头的注意力权重
fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
for i, attention_head in enumerate(attention_heads):
    sns.heatmap(attention_head, annot=True, cmap="viridis", ax=axes[i], xticklabels=range(seq_length), yticklabels=range(seq_length))
    axes[i].set_title(f"Head {i + 1}")

plt.suptitle("Multi-Head Attention Weights")
plt.show()