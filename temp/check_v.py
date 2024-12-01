import torch
import torchvision

from mnn.losses import FocalLoss

x = torch.zeros((2, 10, 6))
y = torch.ones((2, 10, 6))

loss_none = torch.nn.BCELoss(reduction="none")
loss = torch.nn.BCELoss()

m = torch.nn.Sigmoid()

print(loss_none(m(x), m(y)).mean())  # (2, 10, 6)
print(loss(m(x), m(y)))

focal_loss = FocalLoss(gamma=2, alpha=-1, reduction="mean")

print(focal_loss(m(x), m(y)))  # 1.0

print(
    torchvision.ops.sigmoid_focal_loss(x, m(y), alpha=-1, gamma=0.5, reduction="mean")
)  # 1.0
