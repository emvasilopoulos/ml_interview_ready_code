import abc
import time
import yaml

import torch
import os
import random
import cv2
import pathlib

import torch
import torch.utils.tensorboard

import mnn.vision.image_size
import mnn.vision.models.vision_transformer.encoder.config as mnn_encoder_config
import mnn.vision.config as mnn_config
import mnn.vision.dataset.object_detection.preprocessing


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


def prepare_validation_image(
    validation_image_path: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
):

    preprocessor = (
        mnn.vision.dataset.object_detection.preprocessing.ObjectDetectionPreprocessing
    )
    img = cv2.imread(validation_image_path.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padding_percent = random.random()
    val_img_tensor = preprocessor.cv2_image_to_tensor(img)
    val_img_tensor = preprocessor.preprocess_image(
        val_img_tensor,
        expected_image_size,
        padding_percent=padding_percent,
    )
    return val_img_tensor.unsqueeze(0)


def write_validation_image_with_predicted_mask(
    temp_out: torch.Tensor, validation_image: torch.Tensor, id_: str
):
    validation_img = validation_image.squeeze(0).detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    temp_out = temp_out.squeeze(0).detach().cpu()

    image = (validation_img.numpy() * 255).astype("uint8")
    mask = (temp_out.numpy() * 255).astype("uint8")
    ret, thresh1 = cv2.threshold(mask, 51, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(image, image, mask=thresh1)

    os.makedirs("validation_images", exist_ok=True)
    # cv2.imwrite(f"validation_images/image_{id_}.jpg", image)
    cv2.imwrite(f"validation_images/mask_{id_}.jpg", mask)
    cv2.imwrite(f"validation_images/validation_image_{id_}.jpg", masked_image)


class BaseIOTransform:

    @abc.abstractmethod
    def transform_input(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def transform_output(self, batch: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def update_transform_configuration(self) -> None:
        pass


def calculate_iou_batch(
    preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6
):
    """
    Calculate the Intersection over Union (IoU) for a batch of binary masks.

    Args:
        preds (torch.Tensor): Predicted masks of shape (N, H, W) with values in [0, 1].
        targets (torch.Tensor): Ground truth masks of shape (N, H, W) with values in {0, 1}.
        smooth (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU score for each image in the batch.
    """
    # Convert predictions to binary (0 or 1)
    preds = (preds > 0.2).float()

    # Calculate intersection and union for each image in the batch
    intersection = (preds * targets).sum(dim=(1, 2))
    total = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

    union = total - intersection

    # Calculate IoU for each image
    iou = (intersection + smooth) / (
        union + smooth
    )  # Adding smooth to avoid division by zero
    return iou


def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    current_epoch: int,
    io_transform: BaseIOTransform = None,
    prediction_transform: BaseIOTransform = None,
    device: torch.device = torch.device("cpu"),
    validation_image_path: pathlib.Path = None,
    writer: torch.utils.tensorboard.SummaryWriter = None,
) -> None:
    """
    RTX A2000 - i5 7th Gen - 8GB RAM
    forward pass Time taken: 0.4396486282348633 seconds
    backward propagation Time taken: 0.9111251831054688 seconds
    writer Time taken: 0.02442193031311035 seconds
    loss Time taken: 0.0001838207244873047 seconds
    iou Time taken: 0.0010769367218017578 seconds
    """

    model.train()  # important for batch normalization and dropout layers
    running_loss = 0
    running_iou = 0
    log_rate = 1000
    validation_image = prepare_validation_image(
        validation_image_path, model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)
    val_counter = 0
    for i, (image_batch, target0) in enumerate(train_loader):
        image_batch = image_batch.to(
            device=device, dtype=hyperparameters_config.floating_point_precision
        )
        target0 = target0.to(
            device=device, dtype=hyperparameters_config.floating_point_precision
        )

        if io_transform is not None:
            image_batch = io_transform.transform_input(image_batch)
            target0 = io_transform.transform_output(target0)
            io_transform.update_transform_configuration()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass
        output = model(image_batch)
        if prediction_transform is not None:
            output = prediction_transform.transform_output(output)
            prediction_transform.update_transform_configuration()

        # Compute the loss and its gradients
        loss = loss_fn(output, target0)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Log metrics
        training_step = i + current_epoch * len(train_loader)
        current_loss = loss.item()
        current_iou = calculate_iou_batch(output, target0).mean().item()
        if writer is not None:
            writer.add_scalar("Loss/train", current_loss, training_step)
            writer.add_scalar(
                "IoU/train",
                current_iou,
                training_step,
            )

        running_loss += current_loss
        running_iou += current_iou
        if i % log_rate == 0:
            print(
                f"Training step {training_step} | Loss: {running_loss / log_rate:.4f} | IoU: {running_iou / log_rate:.4f}",
            )
            running_loss = 0
            running_iou = 0

            # Store validation image to inspect the model's performance
            temp_out = model(validation_image)
            write_validation_image_with_predicted_mask(
                temp_out, validation_image, f"epoch_{current_epoch}_id_{val_counter}"
            )
            val_counter += 1


def val_once(
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    current_epoch: int,
    io_transform: BaseIOTransform = None,
    prediction_transform: BaseIOTransform = None,
    device: torch.device = torch.device("cpu"),
    writer: torch.utils.tensorboard.SummaryWriter = None,
):
    log_rate = 1000
    model.eval()
    with torch.no_grad():
        running_loss = 0
        running_iou = 0
        for i, (image_batch, target0) in enumerate(val_loader):
            image_batch = image_batch.to(
                device=device, dtype=hyperparameters_config.floating_point_precision
            )
            target0 = target0.to(
                device=device, dtype=hyperparameters_config.floating_point_precision
            )

            if io_transform is not None:
                image_batch = io_transform.transform_input(image_batch)
                target0 = io_transform.transform_output(target0)
                io_transform.update_transform_configuration()

            output = model(image_batch)
            if prediction_transform is not None:
                output = prediction_transform.transform_output(output)
                prediction_transform.update_transform_configuration()

            loss = loss_fn(output, target0)

            current_loss = loss.item()
            current_iou = calculate_iou_batch(output, target0).mean().item()
            validation_step = i + current_epoch * len(val_loader)
            if writer is not None:
                writer.add_scalar("Loss/val", current_loss, validation_step)
                writer.add_scalar(
                    "IoU/val",
                    current_iou,
                    i + current_epoch * len(val_loader),
                )

            running_loss += current_loss
            running_iou += current_iou
            if i % log_rate == 0:
                print(
                    f"Validation step {validation_step}, loss: {running_loss / log_rate:.4f} IoU: {running_iou / log_rate:.4f}"
                )
                running_loss = 0
                running_iou = 0
