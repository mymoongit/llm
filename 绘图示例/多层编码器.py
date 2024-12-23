'''
展示整个编码器的流程，包括嵌入层和多个编码器层。
第一个子图是输入序列，后续子图展示了经过每层编码器后的输出。
'''


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# 示例数据
d_model = 8
seq_length = 5
num_heads = 2




# 模拟输入数据
x = np.random.rand(seq_length, d_model)
num_layers = 3

# 模拟每层的输出
layer_outputs = [np.random.rand(seq_length, d_model) for _ in range(num_layers)]

# 绘制整个 Transformer Encoder 过程
fig, axes = plt.subplots(1, num_layers + 1, figsize=(18, 6))
sns.heatmap(x, annot=True, cmap="Blues", ax=axes[0])
axes[0].set_title("Input Sequence")

for i, layer_output in enumerate(layer_outputs):
    sns.heatmap(layer_output, annot=True, cmap="Purples", ax=axes[i + 1])
    axes[i + 1].set_title(f"Layer {i + 1} Output")

plt.suptitle("Transformer Encoder Layers")
plt.show()