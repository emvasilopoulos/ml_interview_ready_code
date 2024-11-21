import torch
import mnn.vision.models.cnn.components.conv_blocks_down as mnn_conv_blocks_down
import mnn.vision.models.cnn.components.base as mnn_base

class ConvUpBlock(mnn_base.MNNConv):

    def __init__(self, in_channels: int):
        super().__init__(in_channels, in_channels)
        self.cv_same = mnn_conv_blocks_down.ConvBn(
            in_channels, in_channels, kernel=3, stride=1, padding=1
        )

        self.cv_wide = mnn_conv_blocks_down.ConvBn(
            in_channels, in_channels, kernel=3, stride=1, padding=1
        )

        self.cv_wide_tall = mnn_conv_blocks_down.ConvBn(
            in_channels, in_channels, kernel=3, stride=1, padding=1
        )

    def forward(self, x):
        x0 = self.cv_same(x)
        x0_x = torch.cat([x0, x], dim=3)
        x1 = self.cv_wide(x0_x)
        x1_x = torch.cat([x1, x0_x], dim=2)
        x2 = self.cv_wide_tall(x1_x)
        return x2


if __name__ == "__main__":
    x = torch.rand(1, 3, 72, 72)
    cv_up = ConvUpBlock(3)
    print(cv_up(x).shape)
