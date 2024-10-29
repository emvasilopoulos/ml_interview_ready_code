import torch

loss = torch.nn.BCELoss()

x = torch.zeros((1, 480, 640))
x[0, 0, 0] = 1
y = torch.zeros((1, 480, 640))

print(loss(x, y))
print(loss(y, x))

y[0, 0, 1] = 1
print(loss(x, y))

y[0, 0, 1] = 0 # reset
y[0, 0, 2] = 1
print(loss(x, y))

y[0, 0, 2] = 0 # reset
y[0, 0, 3] = 1
print(loss(x, y))
