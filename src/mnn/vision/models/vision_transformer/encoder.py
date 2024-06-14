import dataclasses

import torch
import torch.nn

from mnn.vision.models.vision_transformer.building_blocks import (
    PatchingLayer,
    PatchEmbedder,
    PositionEmbedder,
)

# Vision Transformer Implementation
# https://arxiv.org/pdf/2010.11929
# https://arxiv.org/pdf/2106.14881 # improvements with CNNs


@dataclasses.dataclass
class VisionTranformerImageSize:
    width: int = 640
    height: int = 384
    channels: int = 3


@dataclasses.dataclass
class VisionTransformerEncoderConfiguration:
    use_cnn: bool
    patch_size: int = 32
    hidden_dim: int = 768
    number_of_layers: int = 12
    d_model: int = 384  # TODO - What is this?
    n_heads: int = 6
    feed_forward_dimensions: int = 2048


def transformer_sequence_length(patch_size: int, image_size: VisionTranformerImageSize):
    return (image_size.width // patch_size) * (image_size.height // patch_size)


class VisionTransformerEncoder(torch.nn.Module):
    patch_size: int = 32
    patcher: PatchingLayer
    patch_embedder: PatchEmbedder
    position_embedder: PositionEmbedder
    encoder_layer: torch.nn.TransformerEncoderLayer
    encoder: torch.nn.TransformerEncoder

    def __init__(
        self,
        transformer_encoder_config: VisionTransformerEncoderConfiguration,
        input_image_size: VisionTranformerImageSize,
    ):
        self.input_image_size = input_image_size
        self.transformer_encoder_config = transformer_encoder_config

        self.patcher = PatchingLayer(
            patch_size=self.patch_size,
            image_height=self.input_image_size.height,
            image_width=self.input_image_size.width,
        )
        self.patch_embedder = PatchEmbedder()
        hidden_dim = transformer_encoder_config.hidden_dim
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))

        d_model = transformer_encoder_config.d_model
        n_head = transformer_encoder_config.n_heads
        ff_dim = transformer_encoder_config.feed_forward_dimensions
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff_dim,
            activation="gelu",
        )
        num_layers = transformer_encoder_config.number_of_layers
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )

        self.sequence_length = transformer_sequence_length(
            self.patch_size, input_image_size
        )

    def prepend_xclass_token(self, embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def interpolate_positional_embeddings(
        self, positional_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Reason --> inject inductive bias about 2D structure of the image
        """
        raise NotImplementedError()

    def forward(self, images_batch: torch.Tensor) -> torch.Tensor:
        # Step 1 - Split image(s) into fixed-size patches
        patches_batch = self.patcher(images_batch)

        # Step 2 - flatten patches and map each to embeddings of D dimension
        embeddings_batch = self.patch_embedder(
            patches_batch
        )  # Referred as patch embeddings in the paper
        # Prepend a learnable embedding to the patch embeddings
        embeddings_batch = self.prepend_xclass_token(embeddings_batch)

        # Step 3 - Add positional embeddings to retain positional information
        positional_embeddings_batch = self.position_embedder(embeddings_batch)
        interpolated_embeddings_batch = self.interpolate_positional_embeddings(
            positional_embeddings_batch
        )

        # Step 4 - Pass positional embeddings through the transformer encoder
        transformer_encodings_batch = self.encoder(interpolated_embeddings_batch)

        return transformer_encodings_batch


"""
NOTES
1. It is often beneficial to fine-tune at higher resolution than pre-training

"""

""" HELP WITH IMPLEMENTATION """
# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

if __name__ == "__main__":
    import time

    image = torch.randn(1, 3, 384, 640)
    patch_layer = PatchingLayer(32)
    # Unfold dim 2 == Height to 32x32 & dim 3 == Width to 32x32
    # t0 = time.time()
    # patches = image_to_patches(image)
    # t1 = time.time()
    # print(f"naive image to patches:", t1 - t0, patches.shape)

    t0 = time.time()
    patches = patch_layer(image)
    t1 = time.time()
    print(f"patch layer:", t1 - t0, patches.shape)
