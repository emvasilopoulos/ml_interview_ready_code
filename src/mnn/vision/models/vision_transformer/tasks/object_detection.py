from typing import Tuple
import dataclasses

import torch

import mnn.vision.models.vision_transformer.encoder.config as mnn_encoder_config
import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils


@dataclasses.dataclass
class HeadInputSize:
    sequence_length: int
    embeddings_size: int
    number_of_channels: int


class AddChannelsAndSigmoid(torch.nn.Module):

    def __init__(self, input_size: HeadInputSize):
        super().__init__()

        if input_size.number_of_channels != 3:
            raise ValueError("This head is only compatible with 3 channels input")

        self.add_channels = torch.nn.Conv2d(
            in_channels=input_size.number_of_channels,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_channels(x)
        x = self.sigmoid(x)
        return x


class AddChannelsAndSigmoid2(torch.nn.Module):

    def __init__(self, input_size: HeadInputSize):
        super().__init__()

        if input_size.number_of_channels != 3:
            raise ValueError("This head is only compatible with 3 channels input")

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, 0, :] + x[:, 1, :] + x[:, 2, :]
        x = self.sigmoid(x)
        return x


class ObjectDetectionOrdinalHeadExperimental(torch.nn.Module):

    def __init__(self, input_size: HeadInputSize):
        super().__init__()

        if input_size.number_of_channels == 1:
            self.head = torch.nn.Sigmoid()
        elif input_size.number_of_channels == 3:
            self.head = AddChannelsAndSigmoid2(input_size)


class ObjectDetectionOrdinalHead(torch.nn.Module):

    def __init__(
        self, config: mnn_encoder_config.VisionTransformerEncoderConfiguration
    ):
        super().__init__()

        if config.number_of_layers > 0:
            self.layer = mnn_encoder_utils.get_transformer_encoder_from_config(config)
        else:
            raise ValueError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
