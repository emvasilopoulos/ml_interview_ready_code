import torch

x = torch.zeros((4, 15, 15))

# 3 objects
x[0, -1, 0] = 1
x[0, -1, 1] = 1
x[0, -1, 2] = 1

# 5 objects
x[1, -1, 0] = 1
x[1, -1, 1] = 1
x[1, -1, 2] = 1
x[1, -1, 3] = 1
x[1, -1, 4] = 1

# 0 objects

# 1 object
x[3, -1, 0] = 1


# tensor with number of objects
y = x[:, -1].sum(dim=1)
print(y.shape)
print(y)
