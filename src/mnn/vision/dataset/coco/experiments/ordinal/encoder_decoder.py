import math
from typing import Tuple

import torch
import mnn.vision.image_size
import mnn.vision.process_output.object_detection.grid as mnn_grid


class GridEncoderDecoder:

    def __init__(
        self,
        expected_image_size: mnn.vision.image_size.ImageSize,
        output_shape: mnn.vision.image_size.ImageSize,
        n_classes: int,
    ):
        self.expected_image_size = expected_image_size
        self.output_shape = output_shape
        self.n_classes = n_classes

        image_grid_Sx = int(self.expected_image_size.width ** (1 / 2))
        image_grid_Sy = int(self.expected_image_size.height ** (1 / 2))
        self.image_grid_S = mnn.vision.image_size.ImageSize(
            width=image_grid_Sx, height=image_grid_Sy
        )

        out_grid_Sx = self.output_shape.width ** (1 / 2)
        out_grid_Sy = self.output_shape.height ** (1 / 2)
        self.output_mask_grid_S = mnn.vision.image_size.ImageSize(
            width=out_grid_Sx, height=out_grid_Sy
        )

    def encode(
        self, xc_norm: float, yc_norm: float
    ) -> Tuple[Tuple[float, float], Tuple[int, int]]:

        position_x, position_y = mnn_grid.calculate_grid(
            xc_norm, yc_norm, self.output_shape, self.output_mask_grid_S
        )

        in_grid_x, in_grid_y = mnn_grid.calculate_coordinate_in_grid(
            xc_norm, yc_norm, self.output_shape, self.output_mask_grid_S
        )

        return (in_grid_x, in_grid_y), (position_x, position_y)

    def decode(
        self,
        vector_position: int,
        xc_norm_in_grid_cell: float,
        yc_norm_in_grid_cell: float,
    ) -> Tuple[float, float]:
        """
        Calculate real coordinates based on shape of output tensor
        to get the normalized coordinates
        """
        Sx, Sy = mnn_grid.oneD_position_to_twoD_grid_position(
            vector_position, self.output_mask_grid_S
        )
        xc_norm, yc_norm = mnn_grid.calculate_real_coordinate(
            xc_norm_in_grid_cell,
            yc_norm_in_grid_cell,
            Sx,
            Sy,
            self.output_mask_grid_S,
            self.output_shape,
        )
        return xc_norm, yc_norm


class NoPriorEncoderDecoder:
    def __init__(self, prior_box_shape: mnn.vision.image_size.ImageSize):
        self.prior_box_shape = prior_box_shape
        self.prior_width = prior_box_shape.width
        self.prior_height = prior_box_shape.height

    def encode(self, w_norm: float, h_norm: float) -> Tuple[float, float]:
        return w_norm, h_norm

    def decode(self, w_norm: float, h_norm: float) -> Tuple[float, float]:
        return w_norm, h_norm


def _torch_inverse_sigmoid(x: float) -> float:
    x = torch.Tensor([x])
    return torch.log(x / (1 - x)).item()


class PriorEncoderDecoder:
    eps = 1e-5

    def __init__(self, prior_box_shape: mnn.vision.image_size.ImageSize):
        self.prior_box_shape = prior_box_shape
        self.prior_width = prior_box_shape.width
        self.prior_height = prior_box_shape.height

    # YOLOv3: https://arxiv.org/pdf/1612.08242
    def encode(self, w: float, h: float) -> Tuple[float, float]:
        if 0 < w < 1.0 or 0 < h < 1.0:
            raise ValueError("Width and height should not be normalized.")
        if w == 0:
            w = self.eps
        tw = math.log(w / self.prior_width, math.e)
        if h == 0:
            h = self.eps
        th = math.log(h / self.prior_height, math.e)

        # we require the output to be between 0 and 1
        return (
            torch.sigmoid(torch.Tensor([tw])).item(),
            torch.sigmoid(torch.Tensor([th])).item(),
        )

    def decode(self, tw: float, th: float) -> Tuple[float, float]:
        if tw == 0:
            tw = self.eps
        elif tw == 1:
            tw = 1 - self.eps
        if th == 0:
            th = self.eps
        elif th == 1:
            th = 1 - self.eps
        tw = _torch_inverse_sigmoid(tw)
        th = _torch_inverse_sigmoid(th)

        w = self.prior_width * math.e**tw
        h = self.prior_height * math.e**th
        return w, h
