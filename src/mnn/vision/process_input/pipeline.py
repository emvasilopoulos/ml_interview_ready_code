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


class MyConvertImageDtype(torch.nn.Module):
    """
    Don't use the ready torchvision.transforms.ConvertImageDtype
    because it normalizes the image to [0, 1] with a unique way
    and we want to know the exact way it is normalized before
    feeding it to the model.
    Otherwise changes in the normalization will affect the model's
    performance and it will be hard to understand why.
    """

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.dtype)


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

    dtype_converter = MyConvertImageDtype(torch.float32)
    normalize = mnn.vision.process_input.normalize.basic.NORMALIZE
    pipeline = ProcessInputPipeline(
        dtype_converter=dtype_converter,
        normalize=normalize,
    )

    x = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)

    pipeline_output = pipeline(x)
    deconstructed_pipeline = normalize(dtype_converter(x))

    assert pipeline_output.min() == deconstructed_pipeline.min()
    assert pipeline_output.max() == deconstructed_pipeline.max()
    assert torch.allclose(pipeline_output, deconstructed_pipeline)
