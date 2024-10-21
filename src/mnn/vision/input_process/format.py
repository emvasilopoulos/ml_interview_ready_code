import numpy as np
import numpy.typing as npt
import torch


class FormatConversion:

    @staticmethod
    def cv2_image_to_tensor(x: npt.NDArray[np.uint8]) -> torch.Tensor:
        return torch.from_numpy(x).permute(2, 0, 1).float()
