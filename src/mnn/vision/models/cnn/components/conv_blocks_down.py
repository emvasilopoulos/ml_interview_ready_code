import torch

import mnn.vision.models.cnn.components.base as mnn_base


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


class ConvBn(mnn_base.MNNConv):
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
        activation: torch.nn.Module = torch.nn.SiLU(inplace=True),
    ):
        super().__init__(in_channels, out_channels)
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
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = activation
        self.out_channels = out_channels

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.activation(self.conv(x))


class Bottleneck(mnn_base.MNNConv):
    def __init__(
        self,
        in_channels: int,
        activation: torch.nn.Module = torch.nn.SiLU(inplace=True),
    ):
        hidden_channels = in_channels  # TODO - add logic for hidden channels
        super().__init__(in_channels, in_channels)
        self.conv1 = ConvBn(
            in_channels,
            hidden_channels,
            kernel=3,
            stride=1,
            padding=1,
            activation=activation,
        )
        self.conv2 = ConvBn(
            hidden_channels,
            in_channels,
            kernel=3,
            stride=1,
            padding=1,
            activation=activation,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y


class SPP(mnn_base.MNNConv):
    """
    Spatial Pyramid Pooling
    Modified: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_kernel: int = 5,
        activation: torch.nn.Module = torch.nn.SiLU(inplace=True),
    ):
        super().__init__(in_channels, out_channels)

        hidden_channels = in_channels // 2
        self.conv1 = ConvBn(
            in_channels,
            hidden_channels,
            kernel=1,
            stride=1,
            padding=0,
            activation=activation,
        )
        self.conv2 = ConvBn(
            hidden_channels * 5,
            out_channels,
            kernel=1,
            stride=1,
            padding=0,
            activation=activation,
        )
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2
        )

    def forward(self, x: torch.Tensor):
        y_conv1 = self.conv1(x)
        y_pool1_0 = self.maxpool1(y_conv1)
        y_pool1_1 = self.maxpool1(y_pool1_0)
        y_pool1_2 = self.maxpool1(y_pool1_1)
        y_pool1_3 = self.maxpool1(y_pool1_2)
        y = torch.cat([y_conv1, y_pool1_0, y_pool1_1, y_pool1_2, y_pool1_3], 1)
        return self.conv2(y)


EXPECTED_OUTPUT_RESOLUTION = {
    "same": {
        "kernel": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
    },
    "half": {
        "kernel": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
    },
    "third": {
        "kernel": 3,
        "stride": 3,
        "padding": 1,
        "dilation": 1,
    },
    "quarter": {
        "kernel": 3,
        "stride": 4,
        "padding": 1,
        "dilation": 1,
    },
}
