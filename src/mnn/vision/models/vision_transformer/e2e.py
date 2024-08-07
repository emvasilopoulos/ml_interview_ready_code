import time
import torch
import torch.nn

import mnn.vision.models.vision_transformer.encoder.config as mnn_encoder_config
import mnn.vision.image_size
from mnn.vision.models.vision_transformer.encoder.vit_encoder import (
    RawVisionTransformerRGBEncoder,
    RawVisionTransformerMultiChannelEncoder,
)
from mnn.vision.models.vision_transformer.encoder.vit_encoder_combinators import (
    ThreeChannelsCombinatorToThreeChannels,
    ThreeChannelsCombinator,
)
import mnn.vision.models.vision_transformer.positional_encoders.sinusoidal as mnn_sinusoidal_positional_encoders
import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils


# Experimental Module
class EncoderCombinator(torch.nn.Module):
    def __init__(self, encoder: RawVisionTransformerMultiChannelEncoder):
        super().__init__()
        self.encoder = encoder
        self.combinator = ThreeChannelsCombinatorToThreeChannels(encoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.combinator(x)
        return x


# Experimental Module
class EncoderCombinatorStack(torch.nn.Module):
    def __init__(
        self,
        encoder_config,
        image_size,
        n_high_level_layers: int,
    ):
        super().__init__()
        first_layer = EncoderCombinator(
            RawVisionTransformerRGBEncoder(
                encoder_config,
                image_size,
            )
        )
        hidden_layers = [
            EncoderCombinator(
                RawVisionTransformerMultiChannelEncoder(
                    encoder_config,
                    image_size,
                    in_channels=3,
                    out_channels=3,
                )
            )
            for _ in range(n_high_level_layers - 1)
        ]
        all_layers = [first_layer] + hidden_layers
        self.encoder_combinator_list = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        return self.encoder_combinator_list(x)


class RGBCombinator(torch.nn.Module):
    """
    An RGB image is 3 x 2D matrices. The idea behind this module is to combine the three channels into a single 2D matrix.
    """

    def __init__(self, encoder: RawVisionTransformerMultiChannelEncoder):
        super().__init__()
        self.encoder = encoder
        self.combinator = ThreeChannelsCombinator(encoder)

    def forward(self, x):
        x = self.encoder(x)
        return self.combinator(x)


class MyVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        vit_config: mnn_encoder_config.MyVisionTransformerConfiguration,
        image_size: mnn.vision.image_size.ImageSize,
    ):
        super().__init__()

        self.rgb_combinator = RGBCombinator(
            RawVisionTransformerRGBEncoder(
                vit_config.rgb_combinator_config,
                image_size,
            )
        )
        self.transformer_encoder = (
            mnn_encoder_utils.get_transformer_encoder_from_config(
                vit_config.encoder_config
            )
        )

    def forward(self, x: torch.Tensor):
        x = self.rgb_combinator(x)
        return self.transformer_encoder(x)
