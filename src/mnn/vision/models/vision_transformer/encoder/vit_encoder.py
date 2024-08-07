from typing import Iterable, List

import torch
import torch.nn

from mnn.vision.models.vision_transformer.patchers.unfolder import (
    PatchingLayer,
)
import mnn.vision.image_size
import mnn.vision.models.vision_transformer.embedders.fully_connected as mnn_fc
import mnn.vision.models.vision_transformer.encoder.config as mnn_encoder_config
import mnn.vision.models.vision_transformer.positional_encoders.sinusoidal as mnn_sinusoidal_positional_encoders
import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils

# Vision Transformer Implementation
# https://arxiv.org/pdf/2010.11929
# https://arxiv.org/pdf/2106.14881 # improvements with CNNs


def transformer_sequence_length(
    patch_size: int, image_size: mnn.vision.image_size.ImageSize
):
    return (image_size.width // patch_size) * (image_size.height // patch_size)


# First edition of the Vision Transformer Encoder copying the architecture of the paper
class VisionTransformerEncoder(torch.nn.Module):
    patch_size: int = 32
    patcher: PatchingLayer
    patch_embedder: mnn_fc.FullyConnectedPatchEmbedder
    position_embedder: mnn_sinusoidal_positional_encoders.PositionalEncoding
    encoder_layer: torch.nn.TransformerEncoderLayer

    def __init__(
        self,
        transformer_encoder_config: List[
            mnn_encoder_config.VisionTransformerEncoderConfiguration
        ],
        input_image_size: mnn.vision.image_size.ImageSize,
    ):
        super().__init__()

        self.EXPECTED_INPUT_TENSOR = (
            None,  # batch size
            input_image_size.channels,
            input_image_size.height,
            input_image_size.width,
        )
        self.EXPECTED_OUTPUT_TENSOR = (
            None,  # batch size
            input_image_size.height,
            input_image_size.width,
        )

        self.input_image_size = input_image_size
        self.transformer_encoder_config = transformer_encoder_config
        self._check_matched_config_and_image_size()

        if transformer_encoder_config[0].use_cnn:
            # Simple CNN to extract patches
            self.patcher = torch.nn.Conv2d(
                in_channels=self.input_image_size.channels,
                out_channels=d_model,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            ## TODO - Implement more complex CNNs
        else:
            self.patcher = PatchingLayer(
                patch_size=self.patch_size,
                image_height=self.input_image_size.height,
                image_width=self.input_image_size.width,
            )
            self.patch_embedder = mnn_fc.FullyConnectedPatchEmbedder(
                full_patch_size=self.patch_size
                * self.patch_size
                * self.input_image_size.channels,
                transformer_feature_dimension_size=transformer_encoder_config[
                    0
                ].d_model,
            )

        # Take the hidden_dim of the first encoder from the encoder block
        # to create compatible connection
        d_model = transformer_encoder_config[0].d_model
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.sequence_length = transformer_sequence_length(
            self.patch_size, input_image_size
        )
        self.sequence_length += 1  # Add class token - Don't know why. BERT does it and the paper mentions it

        self.positional_encoder = mnn_sinusoidal_positional_encoders.PositionalEncoding(
            number_of_tokens=self.sequence_length, size_of_token_embedding=d_model
        )

        self.encoder_block = mnn_encoder_utils.get_transformer_encoder_from_config(
            transformer_encoder_config
        )

    def prepend_xclass_token(
        self, batch_size: int, embeddings: torch.Tensor
    ) -> torch.Tensor:
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        sequence_dimension_index = 1
        return torch.cat([batch_class_token, embeddings], dim=sequence_dimension_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = images_batch

        # Step 1 - Split image(s) into fixed-size patches
        x = self.patcher(x)
        # x = patches_batch
        x = x.flatten(start_dim=2)
        # x = patches_batch_flattened

        # Step 2 - flatten patches and map each to embeddings of D dimension
        x = self.patch_embedder(x)  # Referred as patch embeddings in the paper
        # embeddings_batch = patch_embeddings_batch

        # Step 3 - Prepend a learnable embedding to the patch embeddings
        x = self.prepend_xclass_token(batch_size=x.shape[0], embeddings=x)
        # patch_embeddings_with_token_batch = [class_token,patch_embeddings_batch]

        # Step 4 - Add positional embeddings to retain positional information
        x = self.positional_encoder(x)
        # positioned_embeddings_batch = patch_embeddings_with_token_batch + positional_embeddings_batch

        # Step 5 - Pass positional embeddings through the transformer encoder
        # The self attention layer expects inputs in the format (batch_size, seq_length, embedding_size)
        x = self.encoder_block(x)
        # x = transformer_encodings_batch

        # investigate the following from source code of torchvision.VisionTransformer
        """
        The authors of the vision transformer paper state:
        "...we prepend a learnable embedding to the sequence of embedded patches (z0 = xclass),
        whose state at the output of the Transformer encoder (z0L) serves as the image representation y"
        """
        # y = transformer_encodings_batch[
        #     :, 0
        # ]  # so this is the class token after passing through the encoder

        return x

    def _check_matched_config_and_image_size(self):
        if self.input_image_size.width != self.transformer_encoder_config.d_model:
            raise ValueError(
                "The width of the image must be divisible by the patch size"
            )


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
            self.positional_encoder = mnn_sinusoidal_positional_encoders.MyVisionPositionalEncoding(
                number_of_tokens=self.sequence_length,
                size_of_token_embedding=transformer_encoder_config.d_model,
                is_input_normalized=transformer_encoder_config.is_input_to_positional_encoder_normalized,
            )
            modules_layers.append(self.positional_encoder)
        self.encoder_block = mnn_encoder_utils.get_transformer_encoder_from_config(
            transformer_encoder_config
        )
        modules_layers.append(self.encoder_block)
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

    def to_dtype(self, dtype: torch.dtype) -> None:
        for encoder in self.multi_channels_encoder:
            encoder.to(dtype=dtype)
            encoder.position

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
