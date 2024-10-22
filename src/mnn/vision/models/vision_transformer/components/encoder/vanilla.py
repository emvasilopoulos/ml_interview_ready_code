from typing import Iterable, List

import torch
import torch.nn

from mnn.vision.models.vision_transformer.components.patchers.unfolder import (
    PatchingLayer,
)
import mnn.vision.image_size
import mnn.vision.models.vision_transformer.components.embedders.fully_connected as mnn_fc
import mnn.vision.models.vision_transformer.components.encoder.config as mnn_encoder_config
import mnn.vision.models.vision_transformer.components.positional_encoders.sinusoidal as mnn_sinusoidal_positional_encoders
import mnn.vision.models.vision_transformer.components.encoder.utils as mnn_encoder_utils

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
