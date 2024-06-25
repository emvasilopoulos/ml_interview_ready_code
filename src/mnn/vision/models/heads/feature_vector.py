import torch
import torch.nn


class FeatureVectorHead(torch.nn.Module):
    def __init__(self, in_channels: int, feature_vector_size: int):
        """
        Expects a one-dimensional vector as input and returns a one-dimensional vector as output.

        Args:
            in_channels (_type_): _description_
            feature_vector_size (_type_): _description_
        """
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, feature_vector_size)
        # TODO - Activation???

    def forward(self, x):
        return self.fc(x)


class Map2Dto1D(torch.nn.Module):
    def __init__(self, dimension1_size: int, dimension2_size: int):
        super().__init__()
        self.dimension1_size = dimension1_size
        self.dimension2_size = dimension2_size
        self.output_size = dimension1_size * dimension2_size

    def forward(self, x: torch.Tensor):
        if len(x.shape[1:]) > 2:
            raise ValueError("Input tensor must be 2D")
        if x.shape[1] != self.dimension1_size or x.shape[2] != self.dimension2_size:
            raise ValueError(
                f"Input tensor must have shape ({self.dimension1_size}, {self.dimension2_size})"
            )
        return torch.flatten(x, 1)
