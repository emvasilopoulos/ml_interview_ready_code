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
)
import mnn.vision.models.vision_transformer.positional_encoders.sinusoidal as mnn_sinusoidal_positional_encoders


class EncoderCombinator(torch.nn.Module):
    def __init__(self, encoder: RawVisionTransformerMultiChannelEncoder):
        super().__init__()
        self.encoder = encoder
        self.combinator = ThreeChannelsCombinatorToThreeChannels(encoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.combinator(x)
        return x


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


class MyVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        encoder_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
        image_size: mnn.vision.image_size.ImageSize,
        n_high_level_layers: int,
        is_input_normalized: bool,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.positional_encoder = (
            mnn_sinusoidal_positional_encoders.MyVisionPositionalEncoding(
                number_of_tokens=image_size.height,
                size_of_token_embedding=encoder_config.d_model,
                is_input_normalized=is_input_normalized,
                dtype=dtype,
            )
        )

        self.encoder_combinator_list = EncoderCombinatorStack(
            encoder_config,
            image_size,
            n_high_level_layers=n_high_level_layers,
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.to(dtype=dtype)

    def set_batch_size(self, batch_size):
        self.positional_encoder.set_batch_size(batch_size)

    def forward(self, x):
        x = self.positional_encoder(x)
        x = self.encoder_combinator_list(x)
        return x

    def to_dtype(self, dtype: torch.dtype):
        self.positional_encoder.to(dtype=dtype)
        for encoder_combinator in self.encoder_combinator_list:
            encoder_combinator.to(dtype=dtype)
        self.sigmoid.to(dtype=dtype)

    def to_device(self, device: torch.device):
        self.positional_encoder.to(device=device)
        for encoder_combinator in self.encoder_combinator_list:
            encoder_combinator.to(device=device)
        self.sigmoid.to(device=device)
