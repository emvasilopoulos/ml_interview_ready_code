import torch
import numpy as np

import mnn.vision.image_size
import mnn.vision.process_output.object_detection.rectangles_to_mask


class FadingBboxMasks:

    @staticmethod
    def resize_image(x: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = torch.nn.functional.interpolate(
            x,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
        x = x.squeeze(0)
        return x

    @staticmethod
    def pad_image(
        x: torch.Tensor,
        pad_dimension: int,
        expected_dimension_size: int,
        padding_percent: float,
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
        return torch.nn.functional.pad(x, pad_amount_tuple, mode="constant", value=0)

    @staticmethod
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
        return (resize_height, resize_width, pad_dimension, expected_dimension_size)

    @staticmethod
    def adjust_tensor_dimensions(
        x: torch.Tensor,
        expected_image_size: mnn.vision.image_size.ImageSize,
        padding_percent: float = 0,
    ) -> torch.Tensor:
        initial_shape = x.shape
        has_expanded_dim = False
        if len(initial_shape) == 2:
            # Make it work with 2D tensors
            x = x.unsqueeze(0)
            has_expanded_dim = True
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected a 2D tensor or 3D tensor, but got a tensor of shape {x.shape}"
            )

        current_image_size = mnn.vision.image_size.ImageSize(x.shape[2], x.shape[1])
        (
            resize_height,
            resize_width,
            pad_dimension,
            expected_dimension_size_after_pad,
        ) = FadingBboxMasks.calculate_new_tensor_dimensions(
            current_image_size, expected_image_size
        )

        x = FadingBboxMasks.resize_image(x, resize_height, resize_width)
        x = FadingBboxMasks.pad_image(
            x,
            pad_dimension=pad_dimension,
            expected_dimension_size=expected_dimension_size_after_pad,
            padding_percent=padding_percent,
        )

        if (
            x.shape[1] != expected_image_size.height
            or x.shape[2] != expected_image_size.width
        ):
            raise ValueError(
                f"The image was not resized correctly. Initial shape: {initial_shape} New shape: {x.shape} | Expected shape: ({expected_image_size.height}, { expected_image_size.width})"
            )

        if has_expanded_dim:
            x = x.squeeze(0)
        return x

    @staticmethod
    def normalize_image(x: torch.Tensor) -> torch.Tensor:
        return x / 255.0

    @staticmethod
    def preprocess_image(
        x: torch.Tensor,
        expected_image_size: mnn.vision.image_size.ImageSize,
        padding_percent: float = 0,
    ) -> torch.Tensor:
        """
        Expecting tensors of shape (-1, H, W)
        """
        # 1 - Normalization
        x = FadingBboxMasks.normalize_image(x)
        x = FadingBboxMasks.adjust_tensor_dimensions(
            x, expected_image_size, padding_percent=padding_percent
        )
        return x

    @staticmethod
    def bboxes_to_mask(y: torch.Tensor, mask_shape: torch.Size) -> torch.Tensor:
        mask = mnn.vision.process_output.object_detection.rectangles_to_mask.ObjectDetectionOrdinalTransformation.transform_from_normalized_rectangles(
            mask_shape, y
        )
        return mask

    @staticmethod
    def bboxes_to_masks(y: torch.Tensor, mask_shape: torch.Size) -> torch.Tensor:
        return torch.stack(
            [
                mnn.vision.process_output.object_detection.rectangles_to_mask.ObjectDetectionOrdinalTransformation.transform_from_normalized_rectangles(
                    mask_shape, y_i.unsqueeze(0)
                )
                for y_i in y
            ]
        )

    @staticmethod
    def cv2_image_to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).permute(2, 0, 1).float()


if __name__ == "__main__":
    # Unittest preprocess_image
    im_sizes = [
        mnn.vision.image_size.ImageSize(width=1000, height=500),
        mnn.vision.image_size.ImageSize(width=500, height=1000),
        mnn.vision.image_size.ImageSize(width=1000, height=1300),
        mnn.vision.image_size.ImageSize(width=700, height=300),
        mnn.vision.image_size.ImageSize(width=1700, height=1300),
        mnn.vision.image_size.ImageSize(width=1300, height=1700),
    ]
    image_RGBs = [
        torch.rand(3, 300, 400) * 255,
        torch.rand(3, 400, 300) * 255,
        torch.rand(3, 1300, 400) * 255,
        torch.rand(3, 300, 700) * 255,
        torch.rand(3, 1300, 1700) * 255,
        torch.rand(3, 1700, 1300) * 255,
    ]
    for expected_image_size in im_sizes:
        for image_RGB in image_RGBs:
            new_image = FadingBboxMasks.preprocess_image(image_RGB, expected_image_size)
