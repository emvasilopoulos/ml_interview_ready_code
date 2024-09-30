import cv2
import yaml
import pathlib
import time

import torch
import torch.nn

import mnn.vision.image_size
import mnn.vision.models.vision_transformer.encoder.config as mnn_encoder_config
import mnn.vision.config as mnn_config
from mnn.vision.models.vision_transformer.e2e import MyVisionTransformer
from mnn.vision.models.vision_transformer.tasks.object_detection import (
    ObjectDetectionOrdinalHead,
)
import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.loader


def inference_test(image: torch.Tensor, model: torch.nn.Module):
    t0 = time.time()
    output = model(image)
    t1 = time.time()
    print("Time taken:", t1 - t0, "seconds")
    print("Model's output shape:", output.shape)
    traced_model = torch.jit.trace(model.forward, image, check_trace=True, strict=True)
    return traced_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_yaml_file(file_path: pathlib.Path) -> dict:
    with file_path.open(mode="r") as f:
        # Python 3.11 need Loader
        return yaml.load(f, Loader=yaml.FullLoader)


""" CONFIGURATION """


def load_model_config(yaml_path: pathlib.Path):
    model_config_as_dict = read_yaml_file(yaml_path)
    model_config = mnn_encoder_config.MyBackboneVitConfiguration.from_dict(
        model_config_as_dict["network"]["backbone"]
    )
    encoder_config = model_config.encoder_config
    head_config = mnn_encoder_config.VisionTransformerEncoderConfiguration.from_dict(
        model_config_as_dict["network"]["head"]["VisionTransformerHead"]
    )
    return model_config, encoder_config, head_config


def load_hyperparameters_config(yaml_path: pathlib.Path):
    hyperparameters_config_as_dict = read_yaml_file(yaml_path)
    hyperparameters_config = mnn_config.HyperparametersConfiguration.from_dict(
        hyperparameters_config_as_dict
    )
    return hyperparameters_config


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


def val_once(val_loader, model, loss_fn, hyperparameters_config):
    model.eval()
    with torch.no_grad():
        for i, (image_batch, target0) in enumerate(val_loader):
            image_batch = image_batch.to(device)
            prediction_batch = model(image_batch)

            image_batch = image_batch.detach().cpu()
            prediction_batch = prediction_batch.detach().cpu()

            for image, target, prediction in zip(
                image_batch, target0, prediction_batch
            ):
                # convert to numpys for OpenCV
                image = image.permute(1, 2, 0)
                image = (image.numpy() * 255).astype("uint8")
                mask = (target.numpy() * 255).astype("uint8")
                prediction_mask = (prediction.numpy() * 255).astype("uint8")
                # apply mask to image
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                pred_masked_image = cv2.bitwise_and(image, image, mask=prediction_mask)
                cv2.imshow("ground truth masked_image", masked_image)
                cv2.imshow("prediction masked_image", pred_masked_image)

                cv2.waitKey(0)
            break


if __name__ == "__main__":

    model_config, encoder_config, head_config = load_model_config(
        pathlib.Path("model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("hyperparameters.yaml")
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
    object_detection_model.load_state_dict(torch.load("exp1_object_detection.pth"))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    object_detection_model.to(
        device=device, dtype=hyperparameters_config.floating_point_precision
    )

    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/"
    )

    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", object_detection_model.expected_image_size
    )

    train_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hyperparameters_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hyperparameters_config.batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(
        object_detection_model.parameters(), lr=hyperparameters_config.learning_rate
    )
    loss_fn = torch.nn.BCELoss()
    for epoch in range(hyperparameters_config.epochs):

        val_once(val_loader, object_detection_model, loss_fn, hyperparameters_config)
