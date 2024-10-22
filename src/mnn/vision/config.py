import dataclasses
import pathlib

import torch

import mnn.utils
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


def load_hyperparameters_config(yaml_path: pathlib.Path):
    hyperparameters_config_as_dict = mnn.utils.read_yaml_file(yaml_path)
    hyperparameters_config = HyperparametersConfiguration.from_dict(
        hyperparameters_config_as_dict
    )
    return hyperparameters_config
