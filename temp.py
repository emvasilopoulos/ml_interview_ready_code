import torch

loss = torch.nn.BCELoss()
x = torch.zeros(480)
y = torch.zeros(480)

y[0] = 1
y[1] = 0.75
y[2] = 0.5
y[3] = 0.25

print(loss(x, y))

x = torch.zeros((480, 640))
y = torch.zeros((480, 640))
y[0, 0] = 1
y[1, 0] = 0.75
y[2, 0] = 0.5
y[3, 0] = 0.25
print(loss(x, y))
