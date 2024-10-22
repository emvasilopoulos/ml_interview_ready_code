from typing import Tuple
import pathlib

import mnn.vision.models.vision_transformer.components.encoder.config as mnn_encoder_config
import mnn.utils


def load_model_config(
    yaml_path: pathlib.Path,
) -> Tuple[
    mnn_encoder_config.MyBackboneVitConfiguration,
    mnn_encoder_config.VisionTransformerEncoderConfiguration,
    mnn_encoder_config.VisionTransformerEncoderConfiguration,
]:
    model_config_as_dict = mnn.utils.read_yaml_file(yaml_path)
    model_config = mnn_encoder_config.MyBackboneVitConfiguration.from_dict(
        model_config_as_dict["network"]["backbone"]
    )
    encoder_config = model_config.encoder_config
    head_config = mnn_encoder_config.VisionTransformerEncoderConfiguration.from_dict(
        model_config_as_dict["network"]["head"]["VisionTransformerHead"]
    )
    return model_config, encoder_config, head_config
