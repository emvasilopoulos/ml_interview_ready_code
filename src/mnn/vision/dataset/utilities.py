import pathlib
import cv2
import numpy as np
import torch


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image / 255.0


def read_image(path: pathlib.Path) -> torch.Tensor:
    return cv2.imread(path.as_posix())


def load_image_as_tensor(
    image_path: pathlib.Path, normalize: bool, as_bgr: bool
) -> torch.Tensor:
    """
    Summary: Load an image from a file and convert it to a PyTorch tensor.

    Args:
        image_path (pathlib.Path): path to the image file.
        normalize (bool): [0, 255] -> [0, 1] .
        as_bgr (bool): Convert the image to BGR. Otherwise, it will be converted to RGB.

    Returns:
        torch.Tensor: The image as a PyTorch tensor.
    """

    x = read_image(image_path)
    if not as_bgr:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if normalize:
        x = normalize_image(x)
    return torch.tensor(x).permute(2, 0, 1).float()
