from typing import Tuple
import dataclasses

import torch


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


class ObjectDetectionOrdinalHead(torch.nn.Module):

    def __init__(self, input_size: HeadInputSize):
        super().__init__()

        if input_size.number_of_channels == 1:
            self.head = torch.nn.Sigmoid()
        elif input_size.number_of_channels == 3:
            self.head = AddChannelsAndSigmoid2(input_size)
