import torch
import torch.nn as nn
import torch.nn.functional as F

"""
you can use this formula [(W-K+2P)/S]+1
and similarly [(H-K+2P)/S]+1

"""


class ExampleNet(nn.Module):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        image_channels: int = 3,
        n_classes: int = 10,
    ):
        super().__init__()
        w, h, n_ch = image_width, image_height, image_channels
        self.conv1 = nn.Conv2d(
            in_channels=n_ch,
            out_channels=16,
            kernel_size=3,
            padding=0,
            stride=1,
        )
        new_w = (w - 3 + 2 * 0) / 1 + 1
        new_h = (h - 3 + 2 * 0) / 1 + 1
        self.pool = nn.MaxPool2d(2, 2)
        new_w //= 2
        new_h //= 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        new_w = int((new_w - 3 + 2 * 0) // 1 + 1)
        new_h = int((new_h - 3 + 2 * 0) // 1 + 1)
        new_w //= 2
        new_h //= 2
        self.fc1 = nn.Linear(32 * new_w * new_h, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


if __name__ == "__main__":
    net = ExampleNet(100, 120, 3)
    image = torch.randn((1, 100, 120, 3))
    net(image)
    pass
