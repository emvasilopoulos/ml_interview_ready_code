import torch
import torch.nn


def tensor_images_to_patches(
    images: torch.Tensor, patch_size: int = 32
) -> torch.Tensor:
    batch_size, channels, height, width = images.shape
    assert height % patch_size == 0 and width % patch_size == 0
    n_patches = (width * height) // (patch_size**2)

    raise NotImplementedError()


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


class PatchingLayer(torch.nn.Module):
    def __init__(
        self, patch_size: int = 32, image_width: int = 640, image_height: int = 384
    ):
        super().__init__()
        assert image_width % patch_size == 0
        assert image_height % patch_size == 0
        default_pytorch_dilation_value = 1  # Is this what I want?
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(
            kernel_size=patch_size,
            dilation=default_pytorch_dilation_value,
            padding=0,
            stride=patch_size,
        )

    def forward(self, images_batch: torch.Tensor):
        batch_size, n_channels, _, _ = images_batch.shape

        images_batch = self.unfold(images_batch)
        # Reshaping into the shape we want
        a = images_batch.view(
            batch_size, n_channels, self.patch_size, self.patch_size, -1
        ).permute(0, 4, 1, 2, 3)

        return a


class PatchEmbedder(torch.nn.Module):
    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def forward(self, patches_batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class PositionEmbedder(torch.nn.Module):
    def __init__(self):
        # The paper suggests using 1D positional embeddings
        # 2D positional embeddings do not have any advantages

        pass

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
