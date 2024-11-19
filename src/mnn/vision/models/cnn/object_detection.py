import torch

import mnn.vision.models.cnn.components.complex_conv as mnn_complex_conv


class Vanilla(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.image_width = 576
        self.image_height = 576
        self.image_channels = 3

        model = torch.nn.Sequential(
            mnn_complex_conv.ConvBn(
                self.image_channels,
                32,
                kernel=3,
                stride=1,
                padding=1,
                activation=torch.nn.SiLU(),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x


if __name__ == "__main__":
    model = Vanilla()
    x = torch.rand(1, 3, 576, 576)
    y = model(x)
    print(y.shape)
