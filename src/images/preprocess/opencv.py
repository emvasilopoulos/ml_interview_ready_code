import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

# ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# Otherwise calculate mean & std to your own dataset
# Otherwise just use mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]


class OpenCvPreprocessor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        return image / 255

    @staticmethod
    def bgr_to_rbg(image: np.ndarray) -> np.ndarray:
        print(image.shape)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @classmethod
    def to_tensor(cls, image: np.ndarray, device: torch.device) -> torch.tensor:
        image = cls.bgr_to_rbg(image)
        if np.max(image) > 1:
            image = cls.normalize(image)
        return torch.tensor(image, dtype=torch.float32, device=device)

    @classmethod
    def to_tensor_torchvision_way(
        cls, image: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        image = cls.bgr_to_rbg(image)
        return cls.transform(image).to(device)

    @classmethod
    def image_path_to_tensor_torchvision_way(
        cls, image_path: str, device: torch.device
    ):
        return cls.to_tensor_torchvision_way(cv2.imread(image_path), device=device)
