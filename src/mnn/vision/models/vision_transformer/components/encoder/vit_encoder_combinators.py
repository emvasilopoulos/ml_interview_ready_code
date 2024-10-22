import torch
import torch.nn

import mnn.vision.models.vision_transformer.components.encoder.raw_vision_encoder as mnn_vit_encoder


class MultiChannelsCombinator(torch.nn.Module):
    """
    Combines the three channels into one.
    The "RGB pixels" are combined into one "pixel".

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        previous_encoder_block: mnn_vit_encoder.RawVisionTransformerMultiChannelEncoder,
        activation: torch.nn.Module,
    ):
        super().__init__()
        self.n_channels, self.h, self.w = previous_encoder_block.EXPECTED_OUTPUT_TENSOR

        self.layer_w = torch.nn.Linear(
            in_features=self.w, out_features=self.w, bias=True
        )
        self.layer_h = torch.nn.Linear(
            in_features=self.h, out_features=self.h, bias=True
        )
        self.layer_hw = torch.nn.Linear(
            in_features=self.w, out_features=self.w, bias=True
        )
        self.layers_activation = activation
        self.dropout = torch.nn.Dropout(p=0.1)
        self.activation = activation

    def forward(self, previous_encoder_output: torch.Tensor) -> torch.Tensor:
        # branch 1
        x_w = self.layer_w(previous_encoder_output)
        x_w = self.dropout(x_w)
        x_w = self.layers_activation(x_w)

        # branch 2
        x_h = self.layer_h(
            previous_encoder_output.view(-1, self.n_channels, self.w, self.h)
        )
        x_h = self.dropout(x_h)
        x_h = self.layers_activation(x_h)
        x_h = self.layer_hw(x_h.view(-1, self.n_channels, self.h, self.w))
        x_h = self.dropout(x_h)
        x_h = self.layers_activation(x_h)

        # branch 1 + branch 2 + previous_encoder_output (residual connection)
        z = x_w + x_h + previous_encoder_output
        return self.activation(torch.sum(z, dim=1))


class ThreeChannelsCombinator(MultiChannelsCombinator):
    """
    Combines the three channels into one.
    The "RGB pixels" are combined into one "pixel".

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        previous_encoder_block: mnn_vit_encoder.RawVisionTransformerRGBEncoder,
        activation: torch.nn.Module,
    ):
        super().__init__(previous_encoder_block, activation)
        n_channels, _, _ = previous_encoder_block.EXPECTED_OUTPUT_TENSOR
        if n_channels != 3:
            raise ValueError(
                f"Expected 3 channels, got {self.n_channels} channels instead."
            )


class ThreeChannelsCombinatorToThreeChannels(torch.nn.Module):
    """
    Three parallel ThreeChannelsCombinator layers.
    This module re-creates three channels to be passed to the next encoder block.
    """

    def __init__(
        self,
        previous_encoder_block: mnn_vit_encoder.RawVisionTransformerRGBEncoder,
        activation: torch.nn.Module,
    ):
        super().__init__()

        self.to_three_channels = torch.nn.ModuleList(
            [
                ThreeChannelsCombinator(previous_encoder_block, activation)
                for _ in range(3)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_list = [combinator(x).unsqueeze(1) for combinator in self.to_three_channels]
        x = torch.cat(x_list, dim=1)
        return x


class MultiChannelsCombinatorToMultiChannels(torch.nn.Module):
    """
    Three parallel ThreeChannelsCombinator layers.
    This module re-creates three channels to be passed to the next encoder block.
    """

    def __init__(
        self,
        previous_encoder_block: mnn_vit_encoder.RawVisionTransformerMultiChannelEncoder,
        out_channels: int,
        activation: torch.nn.Module,
    ):
        super().__init__()

        self.to_three_channels = torch.nn.ModuleList(
            [
                MultiChannelsCombinator(previous_encoder_block, activation)
                for _ in range(out_channels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_list = [combinator(x).unsqueeze(1) for combinator in self.to_three_channels]
        x = torch.cat(x_list, dim=1)
        return x
