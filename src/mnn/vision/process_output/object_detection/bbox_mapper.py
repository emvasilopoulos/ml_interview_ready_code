from typing import Tuple, NamedTuple

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
