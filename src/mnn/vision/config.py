import dataclasses

import torch

from mnn.torch_utils import FLOATING_POINT_PRECISIONS


@dataclasses.dataclass
class HyperparametersConfiguration:
    batch_size: int
    epochs: int
    optimizer: str
    learning_rate: float
    floating_point_precision: torch.dtype

    @staticmethod
    def from_dict(
        hyperparameters_configuration: dict,
    ) -> "HyperparametersConfiguration":
        return HyperparametersConfiguration(
            hyperparameters_configuration["batch_size"],
            hyperparameters_configuration["epochs"],
            hyperparameters_configuration["optimizer"],
            hyperparameters_configuration["learning_rate"],
            FLOATING_POINT_PRECISIONS[
                hyperparameters_configuration["floating_point_precision"]
            ],
        )
