from typing import Any, Dict

import torch
import numpy as np
import numpy.typing as npt

from mnn.vision.dataset.coco.torch_dataset import COCODatasetInstances2017
import mnn.vision.dataset.object_detection.fading_bboxes_in_mask as mnn_fading_bboxes_in_mask
import mnn.vision.image_size
import mnn.vision.process_input.format
import mnn.vision.process_input.dimensions.pad as mnn_pad
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.process_input.normalize.basic
import mnn.vision.process_output.object_detection
import mnn.vision.process_output.object_detection.rectangles_to_mask


class COCOInstances2017FBM(COCODatasetInstances2017):
    preprocessor = (
        mnn.vision.process_output.object_detection.rectangles_to_mask.ObjectDetectionOrdinalTransformation
    )

    def get_output(
        self,
        img: npt.NDArray[np.uint8],
        annotations: Dict[str, Any],
        padding_percent: float = 0,
    ) -> torch.Tensor:
        bboxes = []
        categories = []
        img_h, img_w = img.shape[0], img.shape[1]

        # Add whole image as a bounding box
        bboxes.append([0, 0, 1, 1])
        for annotation in annotations:
            category = int(annotation["category_id"])
            if (
                self.desired_classes is not None
                and category not in self.desired_classes
            ):
                continue
            x1, y1, w, h = annotation["bbox"]
            bboxes.append([x1 / img_w, y1 / img_h, w / img_w, h / img_h])
            categories.append(annotation["category_id"])

        current_image_size = mnn.vision.image_size.ImageSize(img_w, img_h)
        (
            resize_height,
            resize_width,
            pad_dimension,
            expected_dimension_size_after_pad,
        ) = mnn_resize_fixed_ratio.calculate_new_tensor_dimensions(
            current_image_size, self.expected_image_size
        )

        bboxes_as_mask = mnn_fading_bboxes_in_mask.FadingBboxMasks.bboxes_to_mask(
            torch.Tensor(bboxes).float(), torch.Size((resize_height, resize_width))
        )
        bboxes_as_mask = mnn_pad.pad_image(
            bboxes_as_mask.unsqueeze(0),
            pad_dimension=pad_dimension,
            expected_dimension_size=expected_dimension_size_after_pad,
            padding_percent=padding_percent,
        ).squeeze(0)

        # TODO - support categories somehow
        return bboxes_as_mask

    def get_output_tensor(
        self,
        img: torch.Tensor,
        annotations: Dict[str, Any],
        padding_percent: float = 0,
    ) -> torch.Tensor:
        bboxes = []
        categories = []
        img_h, img_w = img.shape[1], img.shape[2]

        # Add whole image as a bounding box
        bboxes.append([0, 0, 1, 1])
        for annotation in annotations:
            category = int(annotation["category_id"])
            if (
                self.desired_classes is not None
                and category not in self.desired_classes
            ):
                continue
            x1, y1, w, h = annotation["bbox"]
            bboxes.append([x1 / img_w, y1 / img_h, w / img_w, h / img_h])
            categories.append(annotation["category_id"])

        current_image_size = mnn.vision.image_size.ImageSize(img_w, img_h)
        (
            resize_height,
            resize_width,
            pad_dimension,
            expected_dimension_size_after_pad,
        ) = mnn_resize_fixed_ratio.calculate_new_tensor_dimensions(
            current_image_size, self.expected_image_size
        )

        bboxes_as_mask = mnn_fading_bboxes_in_mask.FadingBboxMasks.bboxes_to_mask(
            torch.Tensor(bboxes).float(), torch.Size((resize_height, resize_width))
        )
        bboxes_as_mask = mnn_pad.pad_image(
            bboxes_as_mask.unsqueeze(0),
            pad_dimension=pad_dimension,
            expected_dimension_size=expected_dimension_size_after_pad,
            padding_percent=padding_percent,
        ).squeeze(0)

        # TODO - support categories somehow
        return bboxes_as_mask
