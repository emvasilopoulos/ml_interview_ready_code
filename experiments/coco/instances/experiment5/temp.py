import torch

z = torch.rand(1, 625, 640)
mask = torch.zeros(1, 625)
mask[0, 0] = 1
mask[0, 5] = 1
mask[0, 13] = 1
mask[0, 600] = 1

# Define new_z which is the consists of the elements of z where mask is 1
new_z = z[mask == 1]
print(new_z.shape)
