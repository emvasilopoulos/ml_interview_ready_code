import abc
import time
from typing import Tuple
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
import mnn.vision.dataset.coco.training.metrics as mnn_metrics


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


def load_model_config(
    yaml_path: pathlib.Path,
) -> Tuple[
    mnn_encoder_config.MyBackboneVitConfiguration,
    mnn_encoder_config.VisionTransformerEncoderConfiguration,
    mnn_encoder_config.VisionTransformerEncoderConfiguration,
]:
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

    preprocessor = mnn.vision.dataset.object_detection.preprocessing.FadedBboxMasks
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
    temp_out: torch.Tensor,
    validation_image: torch.Tensor,
    id_: str,
    threshold: int = 127,  # allows values above 50%
):
    validation_img = validation_image.squeeze(0).detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    temp_out = temp_out.squeeze(0).detach().cpu()

    image = (validation_img.numpy() * 255).astype("uint8")

    # reverse mask
    raw_mask = (temp_out.numpy() * 255).astype("uint8")
    ret, mask = cv2.threshold(raw_mask, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    os.makedirs("validation_images", exist_ok=True)
    # cv2.imwrite(f"validation_images/image_{id_}.jpg", image)
    cv2.imwrite(f"validation_images/mask_{id_}.jpg", mask)
    cv2.imwrite(f"validation_images/validation_image_{id_}.jpg", masked_image)
