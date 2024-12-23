'''
展示通过自注意力机制和前馈神经网络处理数据
'''


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# 示例数据
d_model = 8
seq_length = 5
num_heads = 2



# 模拟数据
x = np.random.rand(seq_length, d_model)

# 模拟自注意力输出
attn_output = np.random.rand(seq_length, d_model)

# 模拟前馈网络输出
ff_output = np.random.rand(seq_length, d_model)

# 绘制Transformer Block处理过程
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.heatmap(x, annot=True, cmap="Blues")
plt.title("Input Sequence")

plt.subplot(1, 3, 2)
sns.heatmap(attn_output, annot=True, cmap="Greens")
plt.title("Self-Attention Output")

plt.subplot(1, 3, 3)
sns.heatmap(ff_output, annot=True, cmap="Reds")
plt.title("Feed-Forward Output")

plt.suptitle("Transformer Block Processing")
plt.show()
