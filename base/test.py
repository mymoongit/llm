import torch

# 假设的 scores 和 mask
scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

# 掩蔽未来位置的注意力分数
scores += scores.masked_fill(mask == 0, -1e9)

print(scores)
