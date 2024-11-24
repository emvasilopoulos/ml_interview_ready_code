from typing import Tuple, NamedTuple

import torch

import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.image_size

Bbox = NamedTuple("Bbox", [("x", float), ("y", float), ("w", float), ("h", float)])


def __map_bbox_to_padded_image(
    x1: float,
    y1: float,
    w: float,
    h: float,
    fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
    padding_percent: float,
):
    max_pad_amount = fixed_ratio_components.get_pad_amount()
    pad_amount = int(max_pad_amount * padding_percent)
    if fixed_ratio_components.pad_dimension == 1:
        y1 += pad_amount
        im_w = fixed_ratio_components.resize_width
        im_h = fixed_ratio_components.expected_dimension_size
    elif fixed_ratio_components.pad_dimension == 2:
        x1 += pad_amount
        im_w = fixed_ratio_components.expected_dimension_size
        im_h = fixed_ratio_components.resize_height
    else:
        raise ValueError("The pad_dimension should be 1 or 2")

    if x1 + w >= im_w:
        w = fixed_ratio_components.resize_width - x1 - 1
    if y1 + h >= im_h:
        h = fixed_ratio_components.resize_height - y1 - 1
    return x1, y1, w, h


def translate_norm_bbox_to_padded_image(
    normalized_bbox: Bbox,
    fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
    padding_percent: float,
    field_shape: mnn.vision.image_size.ImageSize,
) -> Tuple[int, int, int, int]:
    if any(x > 1 for x in normalized_bbox):
        raise ValueError("Normalized bounding box values should be between 0 and 1")

    x1, y1, w, h = normalized_bbox

    # Scale - Denormalize
    x1 *= fixed_ratio_components.resize_width
    y1 *= fixed_ratio_components.resize_height
    w *= fixed_ratio_components.resize_width
    h *= fixed_ratio_components.resize_height

    # Translate
    x1, y1, w, h = __map_bbox_to_padded_image(
        x1, y1, w, h, fixed_ratio_components, padding_percent
    )

    # Normalize
    return (
        x1 / field_shape.width,
        y1 / field_shape.height,
        w / field_shape.width,
        h / field_shape.height,
    )


def tl_xywh_to_center_xywh(tlbr: Bbox) -> Bbox:
    x1, y1, w, h = tlbr
    return x1 + w / 2, y1 + h / 2, w, h


def center_xywh_to_tl_xywh(center_xywh: Bbox) -> Bbox:
    x, y, w, h = center_xywh
    x_ = x - w / 2
    if x_ < 0:
        w_out_of_bounds = -x_
        x_ = 0
        w -= w_out_of_bounds

    y_ = y - h / 2
    if y_ < 0:
        h_out_of_bounds = -y_
        y_ = 0
        h -= h_out_of_bounds
    return x_, y_, w, h


def center_xywh_to_tl_xywh_tensor(center_xywh: torch.Tensor) -> torch.Tensor:
    center_xywh[0] = center_xywh[0] - center_xywh[2] / 2
    if center_xywh[0] < 0:
        w_out_of_bounds = -center_xywh[0]
        center_xywh[0] = 0
        center_xywh[2] -= w_out_of_bounds

    center_xywh[1] = center_xywh[1] - center_xywh[3] / 2
    if center_xywh[1] < 0:
        h_out_of_bounds = -center_xywh[1]
        center_xywh[1] = 0
        center_xywh[3] -= h_out_of_bounds

    return center_xywh


def tl_xywh_to_tlbr(tlwh: Bbox) -> Bbox:
    x, y, w, h = tlwh
    return x, y, x + w, y + h


def tl_xywh_to_tlbr_tensor(tlwh: torch.Tensor) -> torch.Tensor:
    tlwh[:, 2] = tlwh[:, 0] + tlwh[:, 2]
    tlwh[:, 3] = tlwh[:, 1] + tlwh[:, 3]
    return tlwh
