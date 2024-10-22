import torch
import torchvision.transforms.functional

import mnn.vision.image_size
from mnn.vision.models.vision_transformer.e2e import MyVisionTransformer
from mnn.vision.models.vision_transformer.tasks.object_detection import (
    ObjectDetectionOrdinalHead,
)
import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.torch_dataset
from mnn.vision.dataset.coco.training.utils import *


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
        self.encoder = MyVisionTransformer(model_config, self.expected_image_size)
        self.head = ObjectDetectionOrdinalHead(config=head_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.head(x.view((x.shape[0], x.shape[2], x.shape[1])))  # swap h,w
        return x.view((x.shape[0], x.shape[2], x.shape[1]))  # reswap h,w


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
    object_detection_model.load_state_dict(torch.load("exp8_object_detection.pth"))
    object_detection_model.to(device=device)

    validation_image = prepare_validation_image(
        validation_image_path, object_detection_model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)

    with torch.no_grad():
        temp_out = object_detection_model(validation_image)
        for i in range(1, 10):
            threshold = int(0.1 * i * 255)
            # temp_out = torch.sigmoid(temp_out)
            write_image_with_mask(
                temp_out,
                validation_image,
                f"test_mask_threshold.{threshold}",
                threshold,
            )
