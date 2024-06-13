import dataclasses
from typing import Callable, List, NamedTuple, Optional

import torch
import torch.nn

# Vision Transformer Implementation
# https://arxiv.org/pdf/2010.11929


@dataclasses.dataclass
class VisionTranformerImageSize:
    width: int = 640
    height: int = 384
    channels: int = 3


def image_to_patches(image: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    batch_size, channels, height, width = image.shape
    assert height % patch_size == 0 and width % patch_size == 0
    n_patches = (width * height) // (patch_size**2)

    # naive patching
    patches = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image[:, :, i : i + patch_size, j : j + patch_size]
            patches.append(patch.unsqueeze(-1))
    return torch.cat(patches, dim=-1)


def tensor_images_to_patches(
    images: torch.Tensor, patch_size: int = 32
) -> torch.Tensor:
    batch_size, channels, height, width = images.shape
    assert height % patch_size == 0 and width % patch_size == 0
    n_patches = (width * height) // (patch_size**2)

    raise NotImplementedError()


class PatchEmbedder(torch.nn.Module):
    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class PositionEmbedder(torch.nn.Module):
    def __init__(self):
        # The paper suggests using 1D positional embeddings
        # 2D positional embeddings do not have any advantages

        pass

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@dataclasses.dataclass
class CnnConfiguration:
    use_cnn: bool


class PatchingLayer(torch.nn.Module):
    def __init__(
        self, patch_size: int = 32, image_width: int = 640, image_height: int = 384
    ):
        super().__init__()
        assert image_width % patch_size == 0
        assert image_height % patch_size == 0

        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, images_batch: torch.Tensor):

        batch_size, n_channels, _, _ = images_batch.shape

        images_batch = self.unfold(images_batch)

        # Reshaping into the shape we want
        a = images_batch.view(
            batch_size, n_channels, self.patch_size, self.patch_size, -1
        ).permute(0, 4, 1, 2, 3)

        return a


class VisionTransformerEncoder(torch.nn.Module):
    patch_size: int = 32
    patcher: PatchingLayer
    patch_embedder: PatchEmbedder
    position_embedder: PositionEmbedder
    encoder_layer: torch.nn.TransformerEncoderLayer
    encoder: torch.nn.TransformerEncoder

    def __init__(self, cnn_config: CnnConfiguration):
        self.cnn_config = cnn_config

        self.patcher = PatchingLayer(patch_size=self.patch_size)
        self.patch_embedder = PatchEmbedder()
        self.encoder_layer = torch.nn.TransformerEncoderLayer()
        self.encoder = torch.nn.TransformerEncoder()

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
        embeddings = self.patch_embedder(
            patches_batch
        )  # Referred as patch embeddings in the paper
        # Prepend a learnable embedding to the patch embeddings
        embeddings = self.prepend_xclass_token(embeddings)

        # Step 3 - Add positional embeddings to retain positional information
        positional_embeddings = self.position_embedder(embeddings)
        interpolated_embeddings = self.interpolate_positional_embeddings(
            positional_embeddings
        )

        # Step 4 - Pass positional embeddings through the transformer encoder
        transformer_encodings = self.encoder(positional_embeddings)


"""
NOTES
1. It is often beneficial to fine-tune at higher resolution than pre-training

"""

if __name__ == "__main__":
    import time

    image = torch.randn(1, 3, 384, 640)
    patch_layer = Patchify(32)
    # Unfold dim 2 == Height to 32x32 & dim 3 == Width to 32x32
    # t0 = time.time()
    # patches = image_to_patches(image)
    # t1 = time.time()
    # print(f"naive image to patches:", t1 - t0, patches.shape)

    t0 = time.time()
    patches = patch_layer(image)
    t1 = time.time()
    print(f"patch layer:", t1 - t0, patches.shape)
