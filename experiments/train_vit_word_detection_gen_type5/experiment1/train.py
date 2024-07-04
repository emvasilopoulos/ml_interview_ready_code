import os
import pathlib
import torch
import yaml

import mnn.vision.image_size
import mnn.vision.models.vision_transformer.encoder.config as mnn_config
import mnn.vision.models.vision_transformer.encoder.block as mnn_encoder_block


def get_current_file_path() -> pathlib.Path:
    return pathlib.Path(os.path.abspath(__file__))


def read_yaml_file(file_path: pathlib.Path) -> dict:
    with file_path.open(mode="r") as f:
        return yaml.load(f)


def get_model_from_configuration(model_configuration: dict) -> torch.nn.Module:
    vit_encoder_config = mnn_config.VisionTransformerEncoderConfiguration(
        model_configuration["use_cnn"],
        model_configuration["patch_size"],
        model_configuration["number_of_layers"],
        model_configuration["d_model"],
        model_configuration["n_heads"],
        model_configuration["feed_forward_dimensions"],
        model_configuration["eps"],
        model_configuration["bias"],
        model_configuration["mask_check"],
    )
    return mnn_encoder_block.TransformerEncoderBlock([vit_encoder_config])


def main() -> None:
    current_file_path = get_current_file_path()
    hyperparameters = read_yaml_file(current_file_path.parent / "hyperparameters.yaml")
    model_configuration = read_yaml_file(current_file_path.parent / "model.yaml")

    input_image_size = mnn.vision.image_size.ImageSize(
        hyperparameters["input_image_size"]["width"],
        hyperparameters["input_image_size"]["height"],
        hyperparameters["input_image_size"]["channels"],
    )
    raw_encoder = get_model_from_configuration(
        model_configuration["VisionTransformerEncoderConfiguration"]
    )
