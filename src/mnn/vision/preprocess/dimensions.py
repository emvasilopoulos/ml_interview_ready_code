import dataclasses
import numpy as np
import numpy.typing as npt
import torch

import mnn.vision.image_size
import mnn.vision.preprocess.base


def resize_image(x: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    has_batch_dimension = len(x.shape) == 4
    if not has_batch_dimension:
        x = x.unsqueeze(0)
    x = torch.nn.functional.interpolate(
        x,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    if not has_batch_dimension:
        x = x.squeeze(0)
    return x


def pad_image(
    x: torch.Tensor,
    pad_dimension: int,
    expected_dimension_size: int,
    padding_percent: float,
    pad_value: int = 0,
) -> torch.Tensor:
    if padding_percent < 0 or padding_percent > 1:
        raise ValueError("The padding_percent should be between 0 and 1")

    # Expecting tensors of shape (3, H, W)
    x_dim_size = x.shape[pad_dimension]
    pad_amount = expected_dimension_size - x_dim_size
    top_pad = int(pad_amount * padding_percent)
    bottom_pad = pad_amount - top_pad
    if pad_dimension == 1:
        pad_amount_tuple = (0, 0, top_pad, bottom_pad)
    elif pad_dimension == 2:
        pad_amount_tuple = (top_pad, bottom_pad, 0, 0)
    else:
        raise ValueError("The pad_dimension should be 1 or 2")
    if pad_amount < 0:
        raise ValueError(
            f"The image is already bigger than the expected height for incoming image: {x.shape}"
        )
    return torch.nn.functional.pad(
        x, pad_amount_tuple, mode="constant", value=pad_value
    )


@dataclasses.dataclass
class ResizeFixedRatioComponents:

    resize_height: int
    resize_width: int
    pad_dimension: int
    expected_dimension_size: int


class ResizeFixedRatio(mnn.vision.preprocess.base.BasePreprocessor):

    def calculate_new_tensor_dimensions(
        current_image_size: mnn.vision.image_size.ImageSize,
        expected_image_size: mnn.vision.image_size.ImageSize,
    ):
        x_w = current_image_size.width
        x_h = current_image_size.height
        expected_image_width = expected_image_size.width
        expected_image_height = expected_image_size.height
        if x_w <= expected_image_width and x_h <= expected_image_height:
            width_ratio = x_w / expected_image_width  # less than 1
            height_ratio = x_h / expected_image_height  # less than 1
            if width_ratio > height_ratio:
                new_height = int(x_h / width_ratio)
                resize_height = new_height
                resize_width = expected_image_width
                pad_dimension = 1
                expected_dimension_size = expected_image_height
            else:
                new_width = int(x_w / height_ratio)
                resize_height = expected_image_height
                resize_width = new_width
                pad_dimension = 2
                expected_dimension_size = expected_image_width
        elif x_w <= expected_image_width and x_h > expected_image_height:
            keep_ratio = x_w / x_h
            new_height = expected_image_height
            new_width = int(new_height * keep_ratio)
            resize_height = expected_image_height
            resize_width = new_width
            pad_dimension = 2
            expected_dimension_size = expected_image_width
        elif x_w > expected_image_width and x_h <= expected_image_height:
            keep_ratio = x_w / x_h
            new_width = expected_image_width
            new_height = int(new_width / keep_ratio)
            resize_height = new_height
            resize_width = expected_image_width
            pad_dimension = 1
            expected_dimension_size = expected_image_height
        else:
            width_ratio = x_w / expected_image_width  # greater than 1
            height_ratio = x_h / expected_image_height  # greater than 1
            if width_ratio > height_ratio:
                new_height = int(x_h / width_ratio)
                resize_height = new_height
                resize_width = expected_image_width
                pad_dimension = 1
                expected_dimension_size = expected_image_height
            else:
                new_width = int(x_w / height_ratio)
                resize_height = expected_image_height
                resize_width = new_width
                pad_dimension = 2
                expected_dimension_size = expected_image_width
        return ResizeFixedRatioComponents(
            resize_height, resize_width, pad_dimension, expected_dimension_size
        )

    def transform(
        x: torch.Tensor,
        expected_image_size: mnn.vision.image_size.ImageSize,
        padding_percent: float = 0.5,
        pad_value: int = 0,
    ) -> torch.Tensor:
        """
        Resize the image to the expected image size while keeping the aspect ratio.

        Args:
            x: The tensor to resize.
            expected_image_size: The expected image size.
            padding_percent: The percentage of padding to add to the image.
            pad_value: The value to pad the image with.
        """

        current_image_size = mnn.vision.image_size.ImageSize(
            width=x.shape[2], height=x.shape[1]
        )
        fixed_ratio_components = ResizeFixedRatio.calculate_new_tensor_dimensions(
            current_image_size, expected_image_size
        )
        x = resize_image(
            x, fixed_ratio_components.resize_height, fixed_ratio_components.resize_width
        )
        x = pad_image(
            x,
            fixed_ratio_components.pad_dimension,
            fixed_ratio_components.expected_dimension_size,
            padding_percent,
            pad_value,
        )
        return x


class FormatConversion:

    @staticmethod
    def cv2_image_to_tensor(x: npt.NDArray[np.uint8]) -> torch.Tensor:
        return torch.from_numpy(x).permute(2, 0, 1).float()
