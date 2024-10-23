import pathlib
from typing import List
import torch
import torchvision

import mnn.vision.process_input.reader


def singleton(cls):
    """Use as decorator of class"""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class ProcessInputPipeline:

    def __init__(
        self,
        dtype_converter: torch.nn.Module,
        normalize: torch.nn.Module,
        modules_list: List[torch.nn.Module] = [],
    ):
        self.__pipeline = torchvision.transforms.Compose(
            [dtype_converter, normalize] + modules_list
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.process_input(x)

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.__pipeline(x)

    def read_and_process_input(self, image_path: pathlib.Path) -> torch.Tensor:
        return self.process_input(
            mnn.vision.process_input.reader.read_image_torchvision(image_path)
        )


if __name__ == "__main__":
    import mnn.vision.process_input.normalize.basic

    pipeline = ProcessInputPipeline(
        dtype_converter=torchvision.transforms.ConvertImageDtype(torch.float32),
        normalize=mnn.vision.process_input.normalize.basic.NORMALIZE,
    )
