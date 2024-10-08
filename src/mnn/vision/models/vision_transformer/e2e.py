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

    def __init__(
        self,
        encoder: RawVisionTransformerMultiChannelEncoder,
        combinator_activation: torch.nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.combinator = ThreeChannelsCombinator(encoder, combinator_activation)

    def forward(self, x):
        x = self.encoder(x)
        return self.combinator(x)


class MyVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        vit_config: mnn_encoder_config.MyBackboneVitConfiguration,
        image_size: mnn.vision.image_size.ImageSize,
    ):
        super().__init__()

        layers = []
        combinator_activation = mnn_encoder_utils.get_combinator_activation_from_config(
            vit_config.rgb_combinator_config
        )
        layers.append(
            RGBCombinator(
                encoder=RawVisionTransformerRGBEncoder(
                    vit_config.rgb_combinator_config,
                    image_size,
                ),
                combinator_activation=combinator_activation,
            )
        )
        if vit_config.encoder_config.number_of_layers > 0:
            layers.append(
                mnn_encoder_utils.get_transformer_encoder_from_config(
                    vit_config.encoder_config
                )
            )

        self.my_vit = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.my_vit(x)


import dataclasses


class DoubleRGBCombinator(torch.nn.Module):
    def __init__(
        self,
        vit_config: mnn_encoder_config.MyBackboneVitConfiguration,
        image_size: mnn.vision.image_size.ImageSize,
    ):
        super().__init__()

        combinator_activation = mnn_encoder_utils.get_combinator_activation_from_config(
            vit_config.rgb_combinator_config
        )

        # Expects images (batch, ch, h, w)
        vit_config_w = vit_config.rgb_combinator_config
        self.rgb_combinator_w = RGBCombinator(
            encoder=RawVisionTransformerRGBEncoder(
                vit_config_w,
                image_size,
            ),
            combinator_activation=combinator_activation,
        )

        # Expects images (batch, ch, w, h)
        vit_config_h = dataclasses.replace(vit_config.rgb_combinator_config)
        temp = vit_config_h.d_model
        vit_config_h.d_model = vit_config_h.feed_forward_dimensions
        vit_config_h.feed_forward_dimensions = temp
        image_size_h = mnn.vision.image_size.ImageSize(
            width=image_size.height, height=image_size.width
        )
        self.rgb_combinator_h = RGBCombinator(
            encoder=RawVisionTransformerRGBEncoder(
                vit_config_h,
                image_size_h,
            ),
            combinator_activation=combinator_activation,
        )

        self.layer_w = torch.nn.Linear(
            in_features=vit_config_w.d_model, out_features=vit_config_w.d_model
        )
        self.layer_h = torch.nn.Linear(
            in_features=vit_config_h.d_model, out_features=vit_config_h.d_model
        )
        self.dropout = torch.nn.Dropout(p=0.1)
        self.activation = combinator_activation

    def forward(self, x: torch.Tensor):
        x_w = self.rgb_combinator_w(x)
        x_w = self.layer_w(x_w)
        x_w = self.dropout(x_w)
        x_w = self.activation(x_w)

        x_h = x.permute(0, 1, 3, 2)
        x_h = self.rgb_combinator_h(x_h)
        x_h = self.layer_h(x_h)
        x_h = self.dropout(x_h)
        x_h = self.activation(x_h).permute(0, 2, 1)

        return self.activation(x_w + x_h)
