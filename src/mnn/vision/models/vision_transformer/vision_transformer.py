import torch
import torch.nn
import mnn.vision.models.vision_transformer.encoder.config as mnn_config
import mnn.vision.image_size
from mnn.vision.models.vision_transformer.encoder.vit_encoder import (
    RawVisionTransformerEncoderRGB,
    ThreeChannelsCombinatorToThreeChannels,
)


class EncoderCombinator(torch.nn.Module):
    def __init__(self, encoder: RawVisionTransformerEncoderRGB):
        super().__init__()
        self.encoder = encoder
        self.combinator = ThreeChannelsCombinatorToThreeChannels(encoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.combinator(x)
        return x


class MyVisionTransformer(torch.nn.Module):
    def __init__(
        self,
        encoder_config: mnn_config.VisionTransformerEncoderConfiguration,
        image_size: mnn.vision.image_size.ImageSize,
        n_high_level_layers: int,
        is_input_normalized: bool,
    ):
        super().__init__()

        self.encoder_combinator_list = torch.nn.Sequential(
            *(
                EncoderCombinator(
                    RawVisionTransformerEncoderRGB(
                        encoder_config,
                        image_size,
                        is_input_normalized=is_input_normalized,
                    )
                )
                for _ in range(n_high_level_layers)
            )
        )
        self.sigmoid = torch.nn.Sigmoid()

    def set_batch_size(self, batch_size):
        for encoder_combinator in self.encoder_combinator_list:
            encoder_combinator.encoder.set_batch_size(batch_size)

    def forward(self, x):

        for encoder_combinator in self.encoder_combinator_list:
            x = encoder_combinator(x)
            x = self.sigmoid(x)
        return x
