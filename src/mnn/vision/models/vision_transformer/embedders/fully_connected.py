import torch
import torch.nn


class FullyConnectedPatchEmbedder(torch.nn.Module):

    def __init__(
        self, full_patch_size: int, transformer_hidden_dimension_size: int
    ) -> None:
        super().__init__()
        self.input_dimension_size = full_patch_size
        self.fc = torch.nn.Linear(
            full_patch_size, transformer_hidden_dimension_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
