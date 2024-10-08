import torch

a = torch.rand((1, 3, 224, 448))

layer = torch.nn.Linear(in_features=448, out_features=556)
print(layer(a).shape)
