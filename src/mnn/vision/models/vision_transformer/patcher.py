import torch
import torch.nn


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
        """

        Args:
            images_batch (torch.Tensor): tensor of shape (batch_size, n_channels, image_height, image_width)

        Returns:
            torch.Tensor: tensor of shape (batch_size, n_patches, n_channels, patch_size, patch_size)
        """

        batch_size, n_channels, _, _ = images_batch.shape

        images_batch = self.unfold(images_batch)

        a = images_batch.view(
            batch_size, n_channels, self.patch_size, self.patch_size, -1
        ).permute(0, 4, 1, 2, 3)

        return a


# naive patching
def image_to_patches(image: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    batch_size, channels, height, width = image.shape
    assert height % patch_size == 0 and width % patch_size == 0
    n_patches = (width * height) // (patch_size**2)

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
