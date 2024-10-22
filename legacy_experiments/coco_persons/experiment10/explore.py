from typing import List
import numpy as np
import torch
import mnn.vision.image_size
from mnn.vision.models.vision_transformer.e2e import (
    MyVisionTransformer,
    DoubleRGBCombinator,
)
from mnn.vision.models.vision_transformer.tasks.object_detection import (
    ObjectDetectionOrdinalHead,
)
import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.torch_dataset


from mnn.vision.dataset.coco.training.utils import *

from mnn.vision.process_output.object_detection.rectangles_to_mask import *


class FocalLoss(torch.nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = "mean"

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.reduce == "sum":
            return torch.sum(loss)
        elif self.reduce == "mean":
            return torch.mean(loss)
        elif self.reduce == "none":
            return loss
        else:
            raise ValueError(
                "The value of the reduce parameter should be either 'sum', 'mean' or 'none'"
            )


class VitObjectDetectionNetwork(torch.nn.Module):

    def __init__(
        self,
        model_config: mnn_encoder_config.MyBackboneVitConfiguration,
        head_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
    ):
        super().__init__()
        expected_image_width = model_config.rgb_combinator_config.d_model
        expected_image_height = (
            model_config.rgb_combinator_config.feed_forward_dimensions
        )
        self.expected_image_size = mnn.vision.image_size.ImageSize(
            width=expected_image_width, height=expected_image_height
        )
        self.encoder = DoubleRGBCombinator(model_config, self.expected_image_size)
        self.head = ObjectDetectionOrdinalHead(config=head_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x)


if __name__ == "__main__":
    model_config, encoder_config, head_config = load_model_config(
        pathlib.Path("model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("hyperparameters.yaml")
    )
    validation_image_path = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/val2017/000000000139.jpg"
    )

    device = torch.device("cuda:0")
    object_detection_model = VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )
    object_detection_model.to(
        device=device,
        dtype=hyperparameters_config.floating_point_precision,
    )
    image_size = object_detection_model.expected_image_size
    validation_image = prepare_validation_image(
        validation_image_path, object_detection_model.expected_image_size
    ).to(
        device=device,
        dtype=hyperparameters_config.floating_point_precision,
    )

    rectangles: List[TopLeftWidthHeightRectangle] = [
        TopLeftWidthHeightRectangle(0, 0, 100, 100),
        TopLeftWidthHeightRectangle(100, 100, 200, 200),
        TopLeftWidthHeightRectangle(200, 200, 300, 300),
    ]

    image_shape = (image_size.height, image_size.width)
    mask = (
        ObjectDetectionOrdinalTransformation.transform(
            image_shape, image_shape, rectangles
        )
        .unsqueeze(0)
        .to(
            device=device,
            dtype=hyperparameters_config.floating_point_precision,
        )
    )
    print(mask.unique())
    mask_cv = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite("mask_cv.jpg", mask_cv)

    focal_loss = FocalLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    output = object_detection_model(validation_image)

    for i in range(20):
        gamma = i * 0.1
        print(f"------------ GAMMA {gamma} ------------")
        focal_loss.gamma = gamma
        for alpha in [0.25, 0.5, 0.75]:
            print(f"-- ALPHA {alpha}", end=" | ")
            focal_loss.alpha = alpha
            loss = focal_loss(output, mask)
            print(f"Loss focal: {loss.item()}")
