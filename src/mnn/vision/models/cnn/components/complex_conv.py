import torch


def autopad(
    kernel: int, padding=None, dilation: int = 1
) -> int:  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if dilation > 1:
        kernel = (
            dilation * (kernel - 1) + 1
            if isinstance(kernel, int)
            else [dilation * (x - 1) + 1 for x in kernel]
        )  # actual kernel-size
    if padding is None:
        padding = (
            kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
        )  # auto-pad
    return padding


class ConvBn(torch.nn.Module):
    """
    modified:
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 1,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        dilation: int = 1,
        activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding=autopad(kernel, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.activation(self.conv(x))


class ConvBnBlock(torch.nn.Module):

    def __init__(
        self, in_channels: int, kernel: int = 3, stride: int = 1, padding: int = 0
    ):

        super().__init__()

        self.block = torch.nn.ModuleList(
            [
                ConvBn(in_channels, 32, kernel=kernel, stride=stride, padding=padding),
                ConvBn(in_channels, 64, kernel=kernel, stride=stride, padding=padding),
                ConvBn(in_channels, 128, kernel=kernel, stride=stride, padding=padding),
            ]
        )

    def forward(self, x):
        xs = [conv(x) for conv in self.block]
        return torch.cat(xs, dim=1)


if __name__ == "__main__":
    model = ConvBnBlock(3)
    x = torch.rand(1, 3, 576, 576)
    y = model(x)
    print(y.shape)
