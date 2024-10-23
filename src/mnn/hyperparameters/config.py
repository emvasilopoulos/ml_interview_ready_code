import dataclasses
import pathlib
from typing import Optional

import torch

import mnn.utils
from mnn.torch_utils import FLOATING_POINT_PRECISIONS


@dataclasses.dataclass
class OptimizerConfiguration:
    name: str
    learning_rate: float
    momentum: float = 0.9
    weight_decay: float = 0.0

    @staticmethod
    def from_dict(optimizer_configuration: dict) -> "OptimizerConfiguration":
        return OptimizerConfiguration(
            optimizer_configuration["name"], optimizer_configuration["learning_rate"]
        )


@dataclasses.dataclass
class HyperparametersConfiguration:
    batch_size: int
    epochs: int
    floating_point_precision: torch.dtype
    optimizer: Optional[OptimizerConfiguration] = None

    @staticmethod
    def from_dict(
        hyperparameters_configuration: dict,
    ) -> "HyperparametersConfiguration":
        optimizer_config = hyperparameters_configuration.get("optimizer", None)
        if optimizer_config:
            optimizer_config = OptimizerConfiguration.from_dict(optimizer_config)
        return HyperparametersConfiguration(
            hyperparameters_configuration["batch_size"],
            hyperparameters_configuration["epochs"],
            FLOATING_POINT_PRECISIONS[
                hyperparameters_configuration["floating_point_precision"]
            ],
            optimizer_config,
        )


def load_hyperparameters_config(yaml_path: pathlib.Path):
    hyperparameters_config_as_dict = mnn.utils.read_yaml_file(yaml_path)
    hyperparameters_config = HyperparametersConfiguration.from_dict(
        hyperparameters_config_as_dict
    )
    return hyperparameters_config
