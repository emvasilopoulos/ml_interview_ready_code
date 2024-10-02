import os
import random
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


def prepare_validation_image():
    import mnn.vision.dataset.object_detection.preprocessing

    preprocessor = (
        mnn.vision.dataset.object_detection.preprocessing.ObjectDetectionPreprocessing
    )
    validation_image = "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/val2017/000000000139.jpg"
    img = cv2.imread(validation_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padding_percent = random.random()
    val_img_tensor = preprocessor.cv2_image_to_tensor(img)
    val_img_tensor = preprocessor.preprocess_image(
        val_img_tensor,
        object_detection_model.expected_image_size,
        padding_percent=padding_percent,
    )
    return val_img_tensor.unsqueeze(0)


def write_validation_image_with_predicted_mask(temp_out, validation_image, id_: str):
    validation_img = validation_image.squeeze(0).detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    temp_out = temp_out.squeeze(0).detach().cpu()

    image = (validation_img.numpy() * 255).astype("uint8")
    mask = (temp_out.numpy() * 255).astype("uint8")
    ret, thresh1 = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(image, image, mask=thresh1)

    os.makedirs("validation_images", exist_ok=True)
    cv2.imwrite(f"validation_images/mask_{id_}.jpg", mask)
    cv2.imwrite(f"validation_images/validation_image_{id_}.jpg", masked_image)


def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: VitObjectDetectionNetwork,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    current_epoch: int,
) -> None:
    model.train()  # important for batch normalization and dropout layers
    running_loss = 0
    log_rate = 100
    validation_image = prepare_validation_image().to(
        device, dtype=hyperparameters_config.floating_point_precision
    )
    val_counter = 0
    for i, (image_batch, target0) in enumerate(train_loader):
        image_batch = image_batch.to(
            device=device, dtype=hyperparameters_config.floating_point_precision
        )
        target0 = target0.to(
            device=device, dtype=hyperparameters_config.floating_point_precision
        )

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass
        output = model(image_batch)

        # Compute the loss and its gradients
        loss = loss_fn(output, target0)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        running_loss += loss.item()
        if i % log_rate == 0:
            print(f"Training Step {i}, loss: {running_loss / log_rate:.4f}")
            running_loss = 0
            temp_out = model(validation_image)
            write_validation_image_with_predicted_mask(
                temp_out, validation_image, f"epoch_{current_epoch}_id_{val_counter}"
            )
            val_counter += 1


def val_once(val_loader, model, loss_fn, hyperparameters_config):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        for i, (image_batch, target0) in enumerate(val_loader):
            image_batch = image_batch.to(
                device=device, dtype=hyperparameters_config.floating_point_precision
            )
            target0 = target0.to(
                device=device, dtype=hyperparameters_config.floating_point_precision
            )
            output = model(image_batch)
            loss = loss_fn(output, target0)
            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Validation step {i}, loss: {running_loss / 10:.4f}")
                running_loss = 0


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

    print(f"Created model with {count_parameters(object_detection_model)}:")
    print(object_detection_model)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    object_detection_model.to(
        device=device, dtype=hyperparameters_config.floating_point_precision
    )
    validation_image = prepare_validation_image().to(
        device, dtype=hyperparameters_config.floating_point_precision
    )
    print(validation_image.shape)
    temp_out = object_detection_model(validation_image)
    print(temp_out.shape)
    exit()
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

    optimizer = torch.optim.Adam(
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
        )
        torch.save(object_detection_model.state_dict(), "exp1_object_detection.pth")
        val_once(val_loader, object_detection_model, loss_fn, hyperparameters_config)
