from typing import List

import torch
import torch.nn

import mnn.vision.models.vision_transformer.encoder.config as mnn_config
import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils


def _validate_config(config: List[mnn_config.VisionTransformerEncoderConfiguration]):
    d_model = config[0].d_model
    for transformer_encoder_config in config:
        if transformer_encoder_config.d_model != d_model:
            raise ValueError("All d_model values must be equal")


## Use with a config list of size 1 at first
## until you learn how to use an encoder
class TransforemerEncoderBlock(torch.nn.Module):
    def __init__(self, config: List[mnn_config.VisionTransformerEncoderConfiguration]):
        super().__init__()
        _validate_config(config)
        self.block = torch.nn.Sequential(
            *[
                mnn_encoder_utils.get_transformer_encoder_from_config(
                    transformer_encoder_config
                )
                for transformer_encoder_config in config
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
