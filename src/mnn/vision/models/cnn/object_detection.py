import time
import torch

import mnn.vision.models.cnn.components.conv_blocks_down as mnn_conv_blocks_down
import mnn.vision.models.cnn.components.conv_blocks_up as mnn_conv_blocks_up


class Vanilla(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.image_width = 576
        self.image_height = 576
        self.image_channels = 3

        """ Down Sampling """
        cv_down0 = mnn_conv_blocks_down.ConvBnBlock2(
            3, kernel=3, stride=1, padding=1, channel_scheme="small"
        )
        cv_down1 = mnn_conv_blocks_down.ConvBnBlock2(
            cv_down0.output_channels,
            kernel=3,
            stride=2,
            padding=1,
            channel_scheme="medium",
        )  # cuts in half --> 288x288
        self.down1 = torch.nn.Sequential(
            cv_down0,
            cv_down1,
        )

        cv_down2 = mnn_conv_blocks_down.ConvBnBlock2(
            cv_down1.output_channels,
            kernel=3,
            stride=1,
            padding=1,
            channel_scheme="medium",
        )
        cv_down3 = mnn_conv_blocks_down.ConvBnBlock2(
            cv_down2.output_channels,
            kernel=3,
            stride=2,
            padding=1,
            channel_scheme="medium",
        )  # cuts in half --> 144x144
        self.down2 = torch.nn.Sequential(
            cv_down2,
            cv_down3,
        )

        cv_down4 = mnn_conv_blocks_down.ConvBnBlock2(
            cv_down3.output_channels,
            kernel=3,
            stride=1,
            padding=1,
            channel_scheme="medium",
        )
        cv_down5 = mnn_conv_blocks_down.ConvBnBlock2(
            cv_down4.output_channels,
            kernel=3,
            stride=2,
            padding=1,
            channel_scheme="small",
        )  # cuts in half --> 72x72
        self.down3 = torch.nn.Sequential(
            cv_down4,
            cv_down5,
            mnn_conv_blocks_down.ConvBn(
                cv_down5.output_channels, 64, kernel=3, stride=1, padding=1
            ),
        )

        """ Up Sampling """
        self.upsample3 = mnn_conv_blocks_up.ConvUpBlock(64)
        after_cat_channels = self.upsample3.output_channels + cv_down3.output_channels

        self.conv_same_3 = mnn_conv_blocks_down.ConvBnBlock2(
            after_cat_channels, kernel=3, stride=1, padding=1, channel_scheme="small"
        )
        self.conv_same_3_1 = mnn_conv_blocks_down.ConvBn(
            self.conv_same_3.output_channels, 128, kernel=3, stride=1, padding=1
        )
        self.conv_same_3_2 = mnn_conv_blocks_down.ConvBn(
            128, 64, kernel=3, stride=1, padding=1
        )

        self.upsample2 = mnn_conv_blocks_up.ConvUpBlock(64)
        after_cat_channels = self.upsample2.output_channels + cv_down1.output_channels
        self.conv_same_2 = mnn_conv_blocks_down.ConvBnBlock2(
            after_cat_channels, kernel=3, stride=1, padding=1, channel_scheme="small"
        )
        self.conv_same_2_1 = mnn_conv_blocks_down.ConvBn(
            self.conv_same_2.output_channels, 64, kernel=3, stride=1, padding=1
        )
        self.conv_same_2_2 = mnn_conv_blocks_down.ConvBn(
            128, 64, kernel=3, stride=1, padding=1
        )

        """ Final Layer """
        cv_final0 = mnn_conv_blocks_down.ConvBnBlock2(64, 1, 1)
        cv_final1_out_channels = 256
        cv_final1 = mnn_conv_blocks_down.ConvBn(
            cv_final0.output_channels,
            out_channels=cv_final1_out_channels,
            kernel=3,
            stride=1,
            padding=1,
        )
        cv_final2 = mnn_conv_blocks_down.ConvBn(
            cv_final1_out_channels,
            out_channels=1,
            kernel=3,
            stride=1,
            padding=1,
        )
        self.final_module = torch.nn.Sequential(
            cv_final0,
            cv_final1,
            cv_final2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down1 = self.down1(x)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x_up2 = self.upsample3(x_down3)
        x_up2 = torch.cat([x_down2, x_up2], dim=1)
        x_up2 = self.conv_same_3(x_up2)
        x_up2 = self.conv_same_3_1(x_up2)
        x_up2 = self.conv_same_3_2(x_up2)

        x_up1 = self.upsample2(x_up2)
        x_up1 = torch.cat([x_down1, x_up1], dim=1)
        x_up1 = self.conv_same_2(x_up1)
        x_up1 = self.conv_same_2_1(x_up1)
        x_up1 = self.conv_same_2_2(x_up1)

        y = self.final_module(x_up1)
        return y.squeeze(dim=1)

    def time_forward(self, x: torch.Tensor) -> torch.Tensor:

        t0 = time.time()
        x_down1 = self.down1(x)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"self.down1: {t1 - t0:.2f} seconds")

        t0 = time.time()
        x_down2 = self.down2(x_down1)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"self.down2: {t1 - t0:.2f} seconds")

        t0 = time.time()
        x_down3 = self.down3(x_down2)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"self.down3: {t1 - t0:.2f} seconds")

        t0 = time.time()
        x_up2 = self.upsample3(x_down3)
        x_up2 = torch.cat([x_down2, x_up2], dim=1)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"up3: {t1 - t0:.2f} seconds")

        t0 = time.time()
        x_up1 = self.upsample2(x_up2)
        x_up1 = torch.cat([x_down1, x_up1], dim=1)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"up2: {t1 - t0:.2f} seconds")

        t0 = time.time()
        y = self.final_module(x_up1)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"up1: {t1 - t0:.2f} seconds")
        return y.squeeze(dim=1)


if __name__ == "__main__":
    import mnn.torch_utils
    import time

    model = Vanilla()
    model.to(device=0)
    with torch.no_grad():
        for _ in range(10):
            t0 = time.time()
            x = torch.rand((4, 3, 576, 576), device="cuda")
            y = model(x)
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"Time: {t1 - t0:.2f} seconds")

        print(y.shape)
        print(
            f"Created model with {mnn.torch_utils.count_parameters(model) / (10 ** 6):.2f} million parameters"
        )
