from typing import Tuple, NamedTuple

import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.image_size

NormBbox = NamedTuple("NormBbox", float, float, float, float)


class COCOAnnotationEncoder:
    def map_bbox_to_padded_image(
        self,
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

    @staticmethod
    def transform_normalized_bbox_to_grid_system(
        normalized_bbox: NormBbox,
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float,
        field_shape: mnn.vision.image_size.ImageSize,
    ) -> Tuple[int, int, int, int]:
        x1_norm, y1_norm, w_norm, h_norm = normalized_bbox

        # Scale - Denormalize
        x1 = x1_norm * fixed_ratio_components.resize_width
        y1 = y1_norm * fixed_ratio_components.resize_height
        w = w_norm * fixed_ratio_components.resize_width
        h = h_norm * fixed_ratio_components.resize_height

        # Translate
        x1, y1, w, h = COCOAnnotationEncoder.map_bbox_to_padded_image(
            x1, y1, w, h, fixed_ratio_components, padding_percent
        )

        # Normalize
        return [
            x1 / field_shape.width,
            y1 / field_shape.height,
            w / field_shape.width,
            h / field_shape.height,
        ]

        pass
