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


import mnn.vision.models.vision_transformer.patchers.unfolder

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
        x = self.head(x)
        return x


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

    object_detection_model = VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )

    print(f"---------- MODEL ARCHITECTURE ------------")
    print(object_detection_model)
    print(
        f"Created model with {count_parameters(object_detection_model) / (10 ** 6)} Mega parameters"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    object_detection_model.to(
        device=device, dtype=hyperparameters_config.floating_point_precision
    )
    validation_image = prepare_validation_image(
        validation_image_path, object_detection_model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)
    temp_out = object_detection_model(validation_image)
    write_validation_image_with_predicted_mask(
        temp_out, validation_image, f"epoch_{0}_id_test"
    )

    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/"
    )

    # See coco["categories"]
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 88]
    train_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir,
        "train",
        object_detection_model.expected_image_size,
        classes=classes,
    )
    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", object_detection_model.expected_image_size, classes=classes
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hyperparameters_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hyperparameters_config.batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(
        object_detection_model.parameters(), lr=hyperparameters_config.learning_rate
    )
    # loss_fn = torch.nn.BCELoss() # This didn't work
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(hyperparameters_config.epochs):
        print(f"---------- EPOCH-{epoch} ------------")
        train_one_epoch(
            train_loader,
            object_detection_model,
            optimizer,
            loss_fn,
            hyperparameters_config,
            epoch,
            io_transform=IOTransform(),
            device=device,
            validation_image_path=validation_image_path,
        )
        torch.save(object_detection_model.state_dict(), "exp3_object_detection.pth")
        val_once(
            val_loader, object_detection_model, loss_fn, hyperparameters_config, device
        )
