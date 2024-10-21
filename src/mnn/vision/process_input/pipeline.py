import pathlib
from typing import List
import torch
import torchvision

import mnn.vision.process_input.reader


class SingletonClass(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance


class ProcessInputPipeline(SingletonClass):

    def __init__(
        self,
        dtype_converter: torch.nn.Module,
        normalize: torch.nn.Module,
        modules_list: List[torch.nn.Module] = [],
    ):
        self.__pipeline = torchvision.transforms.Compose(
            [dtype_converter, normalize] + modules_list
        )

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.__pipeline(x)

    def read_and_process_input(self, image_path: pathlib.Path) -> torch.Tensor:
        return self.process_input(
            mnn.vision.process_input.reader.read_image_torchvision(image_path)
        )

    pass
