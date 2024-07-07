import os
import pathlib
import torch
import yaml

import mnn.vision.image_size
import mnn.vision.models.vision_transformer.encoder.config as mnn_config
import mnn.vision.models.vision_transformer.encoder.block as mnn_encoder_block


def get_current_file_path() -> pathlib.Path:
    return pathlib.Path(os.path.abspath(__file__))
