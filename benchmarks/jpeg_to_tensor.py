import os
import pathlib
import random
import cv2
import PIL.Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def opencv_read_image(image_path: str) -> None:
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def opencv_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.tensor(image).permute(2, 0, 1).float()


def pil_read_image(image_path: str) -> PIL.Image.Image:
    return PIL.Image.open(image_path)


def pil_image_to_tensor(image: PIL.Image.Image) -> torch.Tensor:
    return F.pil_to_tensor(image).float()


def use_pil(image_path: str) -> None:
    return pil_image_to_tensor(pil_read_image(image_path))


def use_opencv(image_path: str) -> None:
    return opencv_image_to_tensor(opencv_read_image(image_path))


images_path = pathlib.Path("../data/image")
tick_pil = cv2.TickMeter()
tick_opencv = cv2.TickMeter()
for i in range(30):
    for image in list(images_path.glob("*.jpeg")):
        tick_pil.start()
        pil_tensor = use_pil(image)
        tick_pil.stop()

        tick_opencv.start()
        opencv_tensor = use_opencv(image)
        tick_opencv.stop()

print("PIL to Tensor:", tick_pil.getAvgTimeMilli(), "ms")
print("OpenCV to Tensor:", tick_opencv.getAvgTimeMilli(), "ms")
print(pil_tensor.shape, opencv_tensor.shape)

ok = True
for i in range(10000):
    random_index_0 = random.randint(0, pil_tensor.shape[0] - 1)
    random_index_1 = random.randint(0, pil_tensor.shape[1] - 1)
    random_index_2 = random.randint(0, pil_tensor.shape[2] - 1)
    if (
        pil_tensor[random_index_0, random_index_1, random_index_2]
        != opencv_tensor[random_index_0, random_index_1, random_index_2]
    ):
        ok = False
        print("Different values at:", random_index_0, random_index_1, random_index_2)
        print("PIL:", pil_tensor[random_index_0, random_index_1, random_index_2])
        print("OpenCV:", opencv_tensor[random_index_0, random_index_1, random_index_2])
        break
print("Same values:", ok)
