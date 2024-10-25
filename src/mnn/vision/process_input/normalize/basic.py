import torch


class BasicNormalize(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0


NORMALIZE = BasicNormalize()
