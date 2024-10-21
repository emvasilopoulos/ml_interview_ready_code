import pathlib

import numpy as np
import numpy.typing as npt
import cv2
import torch
import torchvision
from PIL import Image as PIL_Image


def read_image_torchvision(image_path: pathlib.Path) -> torch.Tensor:
    return torchvision.io.read_image(image_path.as_posix())


def read_image_cv2(image_path: pathlib.Path) -> npt.NDArray[np.uint8]:
    return cv2.imread(image_path.as_posix())


def read_image_pil(image_path: pathlib.Path) -> PIL_Image.Image:
    # can be used with torchvision.transforms.PILToTensor()
    return PIL_Image.open(image_path)
