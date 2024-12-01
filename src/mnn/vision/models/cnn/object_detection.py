import time
import torch

import mnn.vision
import mnn.vision.image_size
import mnn.vision.models.cnn.components.conv_blocks_down as mnn_conv_blocks_down
import mnn.torch_utils

module_timer = mnn.torch_utils.ModuleTimer()


class Vanilla576(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.expected_image_size = mnn.vision.image_size.ImageSize(576, 576)
        layer_output_shape = mnn.vision.image_size.ImageSize(576, 576)
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
        scale_down_factor = 2
        self.down1 = mnn_conv_blocks_down.ConvBn(
            self.same3.out_channels,
            out_channels=128,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
        )  # cuts resolution in half --> 288x288
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor

        # 2 - scale down
        scale_down_factor = 2
        self.down2 = mnn_conv_blocks_down.ConvBn(
            self.down1.out_channels, 256, kernel=3, stride=scale_down_factor, padding=1
        )  # cuts in half --> 144x144
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor

        # bottleneck
        self.down_bootleneck2 = mnn_conv_blocks_down.Bottleneck(self.down2.out_channels)

        # 3 - scale down
        scale_down_factor = 2
        self.down3 = mnn_conv_blocks_down.ConvBn(
            self.down_bootleneck2.out_channels,
            512,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
        )  # cuts in half --> 72x72
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor
        self.down_bootleneck3 = mnn_conv_blocks_down.Bottleneck(self.down3.out_channels)

        """ SPP """
        self.spp = mnn_conv_blocks_down.SPP(
            self.down_bootleneck3.out_channels, out_channels=1024
        )

        scale_down_factor = 3
        self.pre_head0 = mnn_conv_blocks_down.ConvBn(
            self.spp.out_channels,
            out_channels=1024,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
        )  # scale down to (52/scale_down_factor)x(52/scale_down_factor)
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor
        self.pre_head1 = mnn_conv_blocks_down.Bottleneck(self.pre_head0.out_channels)
        self.head0 = mnn_conv_blocks_down.ConvBn(
            self.pre_head1.out_channels,
            out_channels=layer_output_shape.width * layer_output_shape.height,
            kernel=3,
            stride=1,
            padding=1,
            # activation=torch.nn.Sigmoid(),
        )
        self.head1 = torch.nn.Linear(
            self.head0.out_channels, self.head0.out_channels, bias=True
        )
        self.head_activation = torch.nn.Sigmoid()

        self.output_shape = mnn.vision.image_size.ImageSize(
            layer_output_shape.width * layer_output_shape.height,
            layer_output_shape.width * layer_output_shape.height,
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
        x = self.head0(x)
        x = self.head1(x.view(x.shape[0], x.shape[1], -1))
        x = self.head_activation(x)
        return x


class Vanilla(torch.nn.Module):

    def __init__(self, image_size: mnn.vision.image_size.ImageSize):
        super().__init__()
        self.expected_image_size = image_size  # alias -> imsz
        layer_output_shape = mnn.vision.image_size.ImageSize(
            image_size.width, image_size.height
        )
        self.image_channels = self.expected_image_size.channels

        cnn_activation = torch.nn.LeakyReLU
        """ Down Sampling """
        # 1 - scale down
        self.same1 = mnn_conv_blocks_down.ConvBn(
            self.image_channels,
            out_channels=16,
            kernel=3,
            stride=1,
            padding=1,
            activation=cnn_activation(),
        )
        self.same2 = mnn_conv_blocks_down.ConvBn(
            self.same1.out_channels,
            out_channels=32,
            kernel=3,
            stride=1,
            padding=1,
            activation=cnn_activation(),
        )
        self.same3 = mnn_conv_blocks_down.ConvBn(
            self.same2.out_channels,
            out_channels=64,
            kernel=3,
            stride=1,
            padding=1,
            activation=cnn_activation(),
        )
        scale_down_factor = 2
        self.down1 = mnn_conv_blocks_down.ConvBn(
            self.same3.out_channels,
            out_channels=128,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
            activation=cnn_activation(),
        )  # cuts resolution in half --> imsz.width//2 x imsz.height//2
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor

        # 2 - scale down
        scale_down_factor = 2
        self.down2 = mnn_conv_blocks_down.ConvBn(
            self.down1.out_channels,
            256,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
            activation=cnn_activation(),
        )  # cuts in half --> imsz.width//4 x imsz.height//4
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor

        # bottleneck
        self.down_bootleneck2 = mnn_conv_blocks_down.Bottleneck(
            self.down2.out_channels,
            activation=cnn_activation(),
        )

        # 3 - scale down
        scale_down_factor = 2
        self.down3 = mnn_conv_blocks_down.ConvBn(
            self.down_bootleneck2.out_channels,
            512,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
            activation=cnn_activation(),
        )  # cuts in half --> imsz.width//8 x imsz.height//8
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor
        self.down_bootleneck3 = mnn_conv_blocks_down.Bottleneck(
            self.down3.out_channels,
            activation=cnn_activation(),
        )

        """ SPP """
        self.spp = mnn_conv_blocks_down.SPP(
            self.down_bootleneck3.out_channels,
            out_channels=1024,
            activation=cnn_activation(),
        )

        scale_down_factor = 3
        self.pre_head0 = mnn_conv_blocks_down.ConvBn(
            self.spp.out_channels,
            out_channels=1024,
            kernel=3,
            stride=scale_down_factor,
            padding=0,
            activation=cnn_activation(),
        )  # scale down to (imsz.width//8)/scale_down_factor x (imsz.height//8)/scale_down_factor

        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor
        self.pre_head1 = mnn_conv_blocks_down.Bottleneck(
            self.pre_head0.out_channels,
            activation=cnn_activation(),
        )
        self.pre_head2 = mnn_conv_blocks_down.ConvBn(
            self.pre_head1.out_channels,
            out_channels=layer_output_shape.width // 2 * layer_output_shape.height // 2,
            kernel=3,
            stride=1,
            padding=1,
            activation=cnn_activation(),
        )
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        layer_output_shape.width //= 2
        layer_output_shape.height //= 2

        self.head0 = torch.nn.Linear(self.pre_head2.out_channels, 512, bias=True)
        self.head0_activation = torch.nn.LeakyReLU()

        self.head1 = torch.nn.Linear(512, 768, bias=True)
        self.head1_activation = torch.nn.LeakyReLU()

        self.head2 = torch.nn.Linear(768, self.pre_head2.out_channels, bias=True)
        self.head2_activation = torch.nn.Sigmoid()

        self.output_shape = mnn.vision.image_size.ImageSize(
            layer_output_shape.width * layer_output_shape.height,
            layer_output_shape.width * layer_output_shape.height,
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
        x = self.pre_head2(x)
        x = self.max_pool(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.head0(x)
        x = self.head0_activation(x)
        x = self.head1(x)
        x = self.head1_activation(x)
        x = self.head2(x)
        x = self.head2_activation(x)
        return x


class Vanilla576Priors(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.expected_image_size = mnn.vision.image_size.ImageSize(576, 576)
        layer_output_shape = mnn.vision.image_size.ImageSize(576, 576)
        self.n_priors = 9
        self.image_channels = self.expected_image_size.channels

        convs_activation = torch.nn.LeakyReLU
        """ Down Sampling """
        # 1 - scale down
        self.same1 = mnn_conv_blocks_down.ConvBn(
            self.image_channels,
            out_channels=16,
            kernel=3,
            stride=1,
            padding=1,
            activation=convs_activation(),
        )
        self.same2 = mnn_conv_blocks_down.ConvBn(
            self.same1.out_channels,
            out_channels=32,
            kernel=3,
            stride=1,
            padding=1,
            activation=convs_activation(),
        )
        self.same3 = mnn_conv_blocks_down.ConvBn(
            self.same2.out_channels,
            out_channels=64,
            kernel=3,
            stride=1,
            padding=1,
            activation=convs_activation(),
        )
        scale_down_factor = 2
        self.down1 = mnn_conv_blocks_down.ConvBn(
            self.same3.out_channels,
            out_channels=128,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
            activation=convs_activation(),
        )  # cuts resolution in half --> 288x288
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor

        # 2 - scale down
        scale_down_factor = 2
        self.down2 = mnn_conv_blocks_down.ConvBn(
            self.down1.out_channels,
            256,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
            activation=convs_activation(),
        )  # cuts in half --> 144x144
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor

        # bottleneck
        self.down_bootleneck2 = mnn_conv_blocks_down.Bottleneck(
            self.down2.out_channels, activation=convs_activation()
        )

        # 3 - scale down
        scale_down_factor = 2
        self.down3 = mnn_conv_blocks_down.ConvBn(
            self.down_bootleneck2.out_channels,
            512,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
            activation=convs_activation(),
        )  # cuts in half --> 72x72
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor
        self.down_bootleneck3 = mnn_conv_blocks_down.Bottleneck(self.down3.out_channels)

        """ SPP """
        self.spp = mnn_conv_blocks_down.SPP(
            self.down_bootleneck3.out_channels,
            out_channels=1024,
            activation=convs_activation(),
        )

        scale_down_factor = 6
        self.pre_head0 = mnn_conv_blocks_down.ConvBn(
            self.spp.out_channels,
            out_channels=1024,
            kernel=3,
            stride=scale_down_factor,
            padding=1,
            activation=convs_activation(),
        )  # scale down to 12x12
        layer_output_shape.width //= scale_down_factor
        layer_output_shape.height //= scale_down_factor
        self.pre_head1 = mnn_conv_blocks_down.Bottleneck(
            self.pre_head0.out_channels, activation=convs_activation()
        )
        self.head0 = mnn_conv_blocks_down.ConvBn(
            self.pre_head1.out_channels,
            out_channels=layer_output_shape.width * layer_output_shape.height,
            kernel=3,
            stride=1,
            padding=1,
            activation=convs_activation(),
            # activation=torch.nn.Sigmoid(),
        )
        self.heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.head0.out_channels, self.head0.out_channels, bias=True
                )
                for _ in range(self.n_priors)
            ]
        )
        self.head_activation = torch.nn.Sigmoid()

        self.output_shape = mnn.vision.image_size.ImageSize(
            layer_output_shape.width * layer_output_shape.height,
            layer_output_shape.width * layer_output_shape.height,
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
        x = self.head0(x)
        x = torch.stack(
            [head(x.view(x.shape[0], x.shape[1], -1)) for head in self.heads], dim=1
        )
        x = self.head_activation(x)
        return x.permute(0, 2, 1, 3)


if __name__ == "__main__":
    import mnn.torch_utils
    import time

    image_size = mnn.vision.image_size.ImageSize(676, 676)
    model = Vanilla(image_size)
    model.to("cuda")
    for _ in range(3):
        t0 = time.time()
        x = torch.rand((2, 3, image_size.height, image_size.width), device="cuda")
        y = model(x)
        t1 = time.time()
        print(
            "Time taken:",
            t1 - t0,
            "seconds | y-shape:",
            y.shape,
            "model-output:",
            model.output_shape,
        )
