import cv2
import numpy as np
import numpy.typing as npt
import torch


class FormatConversion:

    @staticmethod
    def cv2_image_to_tensor(x: npt.NDArray[np.uint8]) -> torch.Tensor:
        return torch.from_numpy(x).permute(2, 0, 1).float()

    @staticmethod
    def cv2_bgr_to_rbg(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
