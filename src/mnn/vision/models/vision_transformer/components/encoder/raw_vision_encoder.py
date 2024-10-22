import torch
import torch.nn

import mnn.vision.image_size
import mnn.vision.models.vision_transformer.components.encoder.config as mnn_encoder_config
import mnn.vision.models.vision_transformer.components.positional_encoders.sinusoidal as mnn_sinusoidal_positional_encoders
import mnn.vision.models.vision_transformer.components.encoder.utils as mnn_encoder_utils


# Second edition of the Vision Transformer Encoder
class RawVisionTransformerEncoder(torch.nn.Module):
    """
    This layer should process a single channel image.
    """

    def __init__(
        self,
        transformer_encoder_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
        input_image_size: mnn.vision.image_size.ImageSize,
    ) -> None:
        super().__init__()

        self.EXPECTED_INPUT_TENSOR = (
            None,  # batch size
            input_image_size.height,
            input_image_size.width,
        )
        self.EXPECTED_OUTPUT_TENSOR = (
            None,  # batch size
            input_image_size.height,
            input_image_size.width,
        )

        self.input_image_size = input_image_size
        self._check_number_of_channels()
        self.transformer_encoder_config = transformer_encoder_config

        self.sequence_length = (
            self.input_image_size.height
        )  # scanning image from top to bottom

        modules_layers = []
        if transformer_encoder_config.has_positional_encoding:
            positional_encoder = mnn_sinusoidal_positional_encoders.MyVisionPositionalEncoding(
                number_of_tokens=self.sequence_length,
                size_of_token_embedding=transformer_encoder_config.d_model,
                is_input_normalized=transformer_encoder_config.is_input_to_positional_encoder_normalized,
            )
            modules_layers.append(positional_encoder)
        encoder_block = mnn_encoder_utils.get_transformer_encoder_from_config(
            transformer_encoder_config
        )
        modules_layers.append(encoder_block)
        self.raw_vit = torch.nn.Sequential(*modules_layers)

    def _check_number_of_channels(self):
        if self.input_image_size.channels != 1:
            error_message = (
                "The number of channels must be 1, because a Transformer can process"
                "(SequenceLength x EmbeddingSize)"
                "or in other words (Height x Width), not (SequenceLength x EmbeddingSize x Channels)"
            )
            raise ValueError(error_message)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.raw_vit(x)


class RawVisionTransformerMultiChannelEncoder(torch.nn.Module):
    """
    This module should process a multi-channel 2D tensor.
    As a structure, it consists of multiple parallel RawVisionTransformerEncoder modules.
    """

    def __init__(
        self,
        transformer_encoder_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
        input_image_size: mnn.vision.image_size.ImageSize,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        if in_channels != out_channels:
            raise ValueError(
                "The number of input channels must match the number of output channels"
            )

        self.EXPECTED_INPUT_TENSOR = (
            in_channels,
            input_image_size.height,
            input_image_size.width,
        )
        self.EXPECTED_OUTPUT_TENSOR = (
            out_channels,
            input_image_size.height,
            input_image_size.width,
        )

        self.input_image_size = input_image_size
        single_channel_input_image_size = mnn.vision.image_size.ImageSize(
            height=input_image_size.height,
            width=input_image_size.width,
            channels=1,
        )
        self.multi_channels_encoder = torch.nn.ModuleList(
            [
                RawVisionTransformerEncoder(
                    transformer_encoder_config,
                    single_channel_input_image_size,
                )
                for _ in range(out_channels)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_list = [
            encoder(x[:, i]).unsqueeze(1)
            for encoder, i in zip(
                self.multi_channels_encoder, range(x.shape[1]), strict=True
            )
        ]
        x = torch.cat(x_list, dim=1)
        return x


class RawVisionTransformerRGBEncoder(RawVisionTransformerMultiChannelEncoder):

    def __init__(
        self,
        transformer_encoder_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
        input_image_size: mnn.vision.image_size.ImageSize,
    ) -> None:
        super().__init__(
            transformer_encoder_config, input_image_size, in_channels=3, out_channels=3
        )


"""
NOTES
1. It is often beneficial to fine-tune at higher resolution than pre-training
"""

""" HELP WITH IMPLEMENTATION """
# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

if __name__ == "__main__":
    transformer_encoder_config = (
        mnn_encoder_config.VisionTransformerEncoderConfiguration(
            use_cnn=False,
            d_model=224,
        )
    )

    input_image_size = mnn.vision.image_size.ImageSize(
        height=224, width=224, channels=4
    )
    test_tensor = torch.randn((1, 4, 224, 224))
    encoder = RawVisionTransformerMultiChannelEncoder(
        transformer_encoder_config=transformer_encoder_config,
        input_image_size=input_image_size,
        in_channels=4,
        out_channels=4,
    )
    output = encoder(test_tensor)
    print(output.shape)
