from typing import List
import torch


from mnn.vision.process_output.object_detection.rectangles_to_mask import *


from mnn.vision.dataset.coco.training.utils import *
from experiment14.train import *


class CustomBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Calculate BCE
        return -torch.mean(
            y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
        )


if __name__ == "__main__":

    image_shape = torch.Size((960, 540))
    rectangles: List[TopLeftWidthHeightRectangle] = [
        TopLeftWidthHeightRectangle(105, 105, 200, 200),
    ]
    mask = ObjectDetectionOrdinalTransformation.transform(
        image_shape, image_shape, rectangles
    ).unsqueeze(0)
    mask = torch.clamp(mask, min=1e-7, max=1 - 1e-7)

    rectangles: List[TopLeftWidthHeightRectangle] = [
        TopLeftWidthHeightRectangle(100, 100, 200, 200),
    ]
    mask2 = ObjectDetectionOrdinalTransformation.transform(
        image_shape, image_shape, rectangles
    ).unsqueeze(0)

    bce_loss = torch.nn.BCELoss()
    bce_loss2 = CustomBCELoss()

    loss = bce_loss(mask, mask2)
    loss2 = bce_loss2(mask, mask2)
    print(loss, loss2)
