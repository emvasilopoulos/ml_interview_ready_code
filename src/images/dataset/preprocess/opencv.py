import cv2
import numpy as np
import torch
from torchvision import transforms


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
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def to_tensor(self, image: np.ndarray, device: torch.device) -> torch.tensor:
        if max(image) > 1:
            image = self.normalize(image)
        image = self.bgr_to_rbg(image)
        return torch.tensor(image, dtype=torch.float32, device=device)

    def to_tensor_torchvision_way(
        self, image: np.ndarray, device: torch.device
    ) -> torch.Tensor:
        return self.transform(image).to(device)
