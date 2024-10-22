from typing import List
import torch


from mnn.vision.models.heads.object_detection import *
import mnn.vision.dataset.utilities
import mnn.vision.dataset.coco.loader


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


def rect_detection_in_mask(mask: torch.Tensor):
    print("------------------")
    mask = (mask.detach().cpu().numpy() * 255).astype("uint8")
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
        pathlib.Path("experiment14/model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("experiment14/hyperparameters.yaml")
    )

    model = VitObjectDetectionNetwork(model_config, head_config)
    model.load_state_dict(
        torch.load("experiment14/trained_models/exp14_object_detection.pth")
    )

    image_size = model.expected_image_size
    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
    )
    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", image_size, classes=None
    )
    image, mask = val_dataset[0]
    ####
    with torch.no_grad():
        mask_pred = model(image.unsqueeze(0))
    mask_img = rect_detection_in_mask(mask_pred.squeeze(0))
    # Display the result
    cv2.imshow("Detected Rectangles in Prediction", mask_img)
    mask_true = rect_detection_in_mask(mask.squeeze(0))
    cv2.imshow("Detected Rectangles in GT", mask_true)
    cv2.waitKey(0)

    ####
    # rectangles: List[TopLeftWidthHeightRectangle] = [
    #     TopLeftWidthHeightRectangle(100, 100, 100, 100),
    #     TopLeftWidthHeightRectangle(300, 300, 200, 200),
    # ]
    # mask = ObjectDetectionOrdinalTransformation.transform_ground_truth(
    #     torch.Size(image_size.height, image_size.width),
    #     torch.Size(image_size.height, image_size.width),
    #     rectangles,
    # )
