import torch

import mnn.vision.image_size


class GridReshaper(torch.nn.Module):
    def __init__(self, image_size: mnn.vision.image_size.ImageSize):
        """
        We divide the incoming image into a grid of patches.
        We flatten each patch and then concatenate them.
        """

        super().__init__()
        ch, w, h = image_size.channels, image_size.width, image_size.height
        if w != h:
            raise ValueError(
                f"Expected image width and height must be equal. Got {w} and {h}"
            )
        if not w ** (1 / 2).is_integer():
            raise ValueError(
                f"Expected image width and height must be a power of 2. Got {w} and {h}"
            )
        self.input_shape = (ch, h, w)
        self.output_shape = (ch, h, w)
        self.grid_size = int(w ** (1 / 2))

    def forward(self, x: torch.Tensor):
        """
        x: torch.Tensor
            Shape: (batch_size, channels, height, width)
        """
        batch_size, ch, h, w = x.shape
        if (ch, h, w) != self.input_shape:
            raise ValueError(
                f"Expected input shape to be {self.input_shape}. Got {x.shape}"
            )
        x = (
            x.reshape(
                (
                    batch_size,
                    ch,
                    self.grid_size,
                    self.grid_size,
                    self.grid_size,
                    self.grid_size,
                )
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch_size, ch, h, w)
        )
        return x


if __name__ == "__main__":

    x = torch.randn(1, 3, 576, 576)
    reshaper = GridReshaper(mnn.vision.image_size.ImageSize(576, 576, 3))
    y = reshaper(x)
    print(y.shape)
