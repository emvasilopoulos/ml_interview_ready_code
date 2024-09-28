import dataclasses

import torch

from mnn.torch_utils import FLOATING_POINT_PRECISIONS
import mnn.vision.image_size


@dataclasses.dataclass
class HyperparametersConfiguration:
    image_size: mnn.vision.image_size.ImageSize
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
            mnn.vision.image_size.ImageSize.from_dict(
                hyperparameters_configuration["input_image_size"]
            ),
            hyperparameters_configuration["batch_size"],
            hyperparameters_configuration["epochs"],
            hyperparameters_configuration["optimizer"],
            hyperparameters_configuration["learning_rate"],
            FLOATING_POINT_PRECISIONS[
                hyperparameters_configuration["floating_point_precision"]
            ],
        )
