import torch
import torch.nn

import mnn.torch_utils
import mnn.vision.models.vision_transformer.encoder.vit_encoder as mnn_vit_encoder


class MultiChannelsCombinator(torch.nn.Module):
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
        super().__init__()
        n_channels, h, w = previous_encoder_block.EXPECTED_OUTPUT_TENSOR

        self.weights = torch.nn.Parameter(
            data=mnn.torch_utils.initialize_weights(torch.Size([n_channels, h, w]))
        )

        self.activation = activation

    def forward(self, previous_encoder_output: torch.Tensor) -> torch.Tensor:
        # Element-wise multiplication
        x = previous_encoder_output * self.weights
        # Sum along the channel dimension
        x = torch.sum(x, dim=1)
        return self.activation(x)


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

    """
    Broken Down:
    self.weights_r = torch.nn.Parameter(
            torch.randn((previous_block_output_height, previous_block_output_width))
        )
    self.weights_g = torch.nn.Parameter(
        torch.randn((previous_block_output_height, previous_block_output_width))
    )
    self.weights_b = torch.nn.Parameter(
        torch.randn((previous_block_output_height, previous_block_output_width))
    )
    ~~~~~~~~~~~~~~~~~~~
    output_vec = (
        previous_encoder_output[:, 0] * self.weights_r
        + previous_encoder_output[:, 1] * self.weights_g
        + previous_encoder_output[:, 2] * self.weights_b
    )
    """


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
