from typing import List
import numpy as np
import torch


from mnn.vision.models.heads.object_detection import *
import mnn.vision.dataset.utilities
import mnn.vision.dataset.coco.torch_dataset


from mnn.vision.dataset.coco.training.utils import *
from experiment16.train import *


class CustomBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Calculate BCE
        return -torch.mean(
            y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
        )


def torch_mask_to_cv(mask: torch.Tensor) -> np.ndarray:
    return (mask.detach().cpu().numpy() * 255).astype("uint8")


def rect_detection_in_mask(mask: torch.Tensor):
    print("------------------")
    mask = torch_mask_to_cv(mask)
    mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Step 3: Apply edge detection
    edges = cv2.Canny(mask, 0, 255)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Print contours analysis
    print(f"Number of contours: {len(contours)}")
    for contour in contours:
        print(f"Contour shape: {contour.shape}")

    # for i, contour in enumerate(contours):
    #     print(f"Contour {i}: {cv2.contourArea(contour)}")

    cv2.drawContours(mask_img, contours, -1, (0, 255, 0), 3)

    return mask_img


if __name__ == "__main__":
    model_config, encoder_config, head_config = load_model_config(
        pathlib.Path("experiment16/model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("experiment16/hyperparameters.yaml")
    )

    model = VitObjectDetectionNetwork(model_config, head_config)
    model.load_state_dict(
        torch.load("experiment16/trained_models/exp16_object_detection_9epochs.pth")
    )

    image_size = model.expected_image_size
    dataset_dir = pathlib.Path("/home/manos/ml_interview_ready_code/data")
    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", image_size, classes=None
    )
    image, mask_target = val_dataset[1]
    loss_fn = torch.nn.BCELoss()
    threshold_fn = torch.nn.Threshold(0.25, 0)
    for i in range(2):
        ####
        with torch.no_grad():
            mask_pred = threshold_fn(model(image.unsqueeze(0)))

        loss = loss_fn(mask_pred, mask_target.unsqueeze(0)).item()
        print(f"Loss-{i}: {loss}")

        mask_pred_cv = torch_mask_to_cv(mask_pred.squeeze(0))
        cv2.imwrite(
            f"pred_mask-{i}.jpg", cv2.cvtColor(mask_pred_cv, cv2.COLOR_GRAY2BGR)
        )

        mask_target_cv = torch_mask_to_cv(mask_target)
        cv2.imwrite(
            f"target_mask-{i}.jpg", cv2.cvtColor(mask_target_cv, cv2.COLOR_GRAY2BGR)
        )
