import time
import torch

import mnn.vision
import mnn.vision.image_size
import mnn.vision.models.cnn.components.conv_blocks_down as mnn_conv_blocks_down
import mnn.vision.models.cnn.components.conv_blocks_up as mnn_conv_blocks_up
import mnn.torch_utils

module_timer = mnn.torch_utils.ModuleTimer()

class Vanilla576(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.expected_image_size = mnn.vision.image_size.ImageSize(576, 576)
        self.output_shape = mnn.vision.image_size.ImageSize(324, 324)
        self.image_channels = self.expected_image_size.channels

        """ Down Sampling """
        # 1 - scale down
        self.same1 = mnn_conv_blocks_down.ConvBn(
            self.image_channels, out_channels=16, kernel=3, stride=1, padding=1
        )
        self.same2 = mnn_conv_blocks_down.ConvBn(
            self.same1.out_channels, out_channels=32, kernel=3, stride=1, padding=1
        )
        self.same3 = mnn_conv_blocks_down.ConvBn(
            self.same2.out_channels, out_channels=64, kernel=3, stride=1, padding=1
        )
        self.down1 = mnn_conv_blocks_down.ConvBn(
            self.same3.out_channels, out_channels=128, kernel=3, stride=2, padding=1
        ) # cuts resolution in half --> 288x288

        # 2 - scale down
        self.down2 = mnn_conv_blocks_down.ConvBn(
            self.down1.out_channels, 256, kernel=3, stride=2, padding=1
        ) # cuts in half --> 144x144

        # bottleneck
        self.down_bootleneck2 = mnn_conv_blocks_down.Bottleneck(self.down2.out_channels)

        # 3 - scale down
        self.down3 = mnn_conv_blocks_down.ConvBn(
            self.down_bootleneck2.out_channels, 512, kernel=3, stride=2, padding=1
        )
        self.down_bootleneck3 = mnn_conv_blocks_down.Bottleneck(self.down3.out_channels)

        """ SPP """
        self.spp = mnn_conv_blocks_down.SPP(self.down_bootleneck3.out_channels, out_channels=1024)

        self.pre_head0 = mnn_conv_blocks_down.ConvBn(
            self.spp.out_channels, out_channels=1024, kernel=3, stride=4, padding=1
        )
        self.pre_head1 = mnn_conv_blocks_down.Bottleneck(
            self.pre_head0.out_channels
        )
        self.head = mnn_conv_blocks_down.ConvBn(
            self.pre_head1.out_channels, out_channels=324, kernel=3, stride=1, padding=1, activation=torch.nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.same1(x)
        x = self.same2(x)
        x = self.same3(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down_bootleneck2(x)
        x = self.down3(x)
        x = self.down_bootleneck3(x)
        x = self.spp(x)
        x = self.pre_head0(x)
        x = self.pre_head1(x)
        x = self.head(x)
        return x.view(x.shape[0], x.shape[1], -1)


if __name__ == "__main__":
    import mnn.torch_utils
    import time

    model = Vanilla576()
    model.to(device=0)
    for _ in range(100):
        t0 = time.time()
        x = torch.rand((2, 3, 576, 576), device="cuda")
        y = model(x)
        torch.cuda.synchronize()
        t1 = time.time()
        print("Time taken:", t1 - t0, "seconds | shape:", y.shape)
