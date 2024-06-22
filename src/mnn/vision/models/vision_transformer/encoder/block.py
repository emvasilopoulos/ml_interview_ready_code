from typing import List

import torch
import torch.nn

import mnn.vision.models.vision_transformer.encoder.config as mnn_config
import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils


class TransforemerEncoderBlock(torch.nn.Module):
    def __init__(self, config: List[mnn_config.VisionTransformerEncoderConfiguration]):
        super().__init__()
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
