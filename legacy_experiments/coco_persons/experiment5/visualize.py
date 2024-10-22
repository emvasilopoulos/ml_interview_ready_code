import torch
import torchvision.transforms.functional

import mnn.vision.image_size
from mnn.vision.models.vision_transformer.e2e import MyVisionTransformer
from mnn.vision.models.vision_transformer.tasks.object_detection import (
    ObjectDetectionOrdinalHead,
)
import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.loader
from mnn.vision.dataset.coco.training.utils import *


class IOTransform(BaseIOTransform):

    current_angle: int = 0

    def transform_input(self, batch: torch.Tensor) -> torch.Tensor:
        return torchvision.transforms.functional.rotate(batch, self.current_angle)

    def transform_output(self, batch: torch.Tensor) -> torch.Tensor:
        return torchvision.transforms.functional.rotate(batch, -self.current_angle)

    def update_transform_configuration(self) -> None:
        self.current_angle = random.randint(0, 359)


class VitObjectDetectionNetwork(torch.nn.Module):

    def __init__(
        self,
        model_config: mnn_encoder_config.MyBackboneVitConfiguration,
        head_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
    ):
        super().__init__()
        expected_image_width = model_config.encoder_config.d_model
        expected_image_height = model_config.encoder_config.feed_forward_dimensions
        self.expected_image_size = mnn.vision.image_size.ImageSize(
            width=expected_image_width, height=expected_image_height
        )
        self.encoder = MyVisionTransformer(model_config, image_size)
        self.head = ObjectDetectionOrdinalHead(config=head_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torchvision.transforms.functional.rotate(x, 90)
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

    batch_size = hyperparameters_config.batch_size
    embedding_size = model_config.rgb_combinator_config.d_model
    sequence_length = model_config.rgb_combinator_config.feed_forward_dimensions
    image_size = mnn.vision.image_size.ImageSize(
        width=embedding_size, height=sequence_length
    )

    hidden_dim = embedding_size
    image_RGB = torch.rand(batch_size, 3, image_size.height, image_size.width) * 255
    image_RGB = image_RGB.to(device=torch.device("cuda:0"))
    object_detection_model = VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )
    object_detection_model = object_detection_model.to(device=torch.device("cuda:0"))

    # visualize the model
    import mnn.visualize

    mnn.visualize.save_model_graph_as_png(
        object_detection_model, image_RGB, pathlib.Path("model.png")
    )
