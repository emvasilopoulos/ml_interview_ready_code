import abc
import pathlib
import torch
import torchvision


class SingletonClass(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SingletonClass, cls).__new__(cls)
        return cls.instance


class BasePreprocessor(SingletonClass):

    def __init__(self, normalize: torchvision.transforms.Normalize):
        self.__read_pipeline = torchvision.transforms.Compose(
            [
                torchvision.transforms.ConvertImageDtype(torch.float32),
                normalize,
            ]
        )

    def read_image(self, image_path: pathlib.Path) -> torch.Tensor:
        return self.__read_pipeline(torchvision.io.read_image(image_path.as_posix()))
