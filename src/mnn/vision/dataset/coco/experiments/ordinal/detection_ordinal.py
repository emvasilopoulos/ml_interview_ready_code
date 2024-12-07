import os
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import cv2

from mnn.vision.dataset.coco.experiments.ordinal.base import (
    BaseCOCOInstances2017Ordinal,
)
import mnn.vision.dataset.coco.experiments.ordinal.encoder_decoder as mnn_vector_encoder
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.image_size
import mnn.vision.process_output.object_detection.bbox_mapper as mnn_bbox_mapper
import mnn.vision.process_output.object_detection.grid as mnn_grid

import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)

CATEGORY_FREQUENCIES = {
    0: 7,
    1: 7,
    2: 7,
    3: 6,
    4: 5,
    5: 4,
    6: 3,
    7: 3,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 2,
    18: 2,
    19: 2,
    20: 2,
    21: 2,
    22: 2,
    23: 2,
    24: 2,
    25: 2,
    26: 2,
    27: 2,
    28: 2,
    29: 2,
    30: 2,
    31: 2,
    32: 2,
    33: 2,
    34: 2,
    35: 2,
    36: 2,
    37: 2,
    38: 2,
    39: 2,
    40: 2,
    41: 2.5,
    42: 2.5,
    43: 2.5,
    44: 2.5,
    45: 2.5,
    46: 2.5,
    47: 2.5,
    48: 2.5,
    49: 2.5,
    50: 2.5,
    51: 2.5,
    52: 2.5,
    53: 2.5,
    54: 2.5,
    55: 2.5,
    56: 2.5,
    57: 2.5,
    58: 2.5,
    59: 2.5,
    60: 2.5,
    61: 2.4,
    62: 2.3,
    63: 2.2,
    64: 2.1,
    65: 2.0,
    66: 1.9,
    67: 1.8,
    68: 1.7,
    69: 1.6,
    70: 1.5,
    71: 1.4,
    72: 1.3,
    73: 1.2,
    74: 1.1,
    75: 1.0,
    76: 0.9,
    77: 0.8,
    78: 0.7,
    79: 0.6,
    80: 0.5,
    81: 0.5,
    82: 0.5,
    83: 0.5,
    84: 0.5,
    85: 0.5,
    86: 0.5,
    87: 0.5,
    88: 0.5,
    89: 0.5,
    90: 0.5,
    91: 0.5,
    92: 0.5,
    93: 0.5,
    94: 0.5,
    95: 0.5,
    96: 0.5,
    97: 0.5,
    98: 0.5,
    99: 0.5,
}


def get_balanced_category_weight(category: int):
    return 1 / CATEGORY_FREQUENCIES[category]


class COCOInstances2017Ordinal(BaseCOCOInstances2017Ordinal):

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        expected_image_size: mnn.vision.image_size.ImageSize,
        output_shape: mnn.vision.image_size.ImageSize = None,
    ):
        self.N_CLASSES = 79
        super().__init__(
            data_dir, split, expected_image_size, output_shape=output_shape
        )
        if self.output_shape.height ** (1 / 2) % 1 != 0:
            raise ValueError(
                f"The square root of the height of the output shape for '{__class__}' should be an integer. Current value: {self.output_shape.height}"
            )

        self.grid_encoder_decoder = mnn_vector_encoder.GridEncoderDecoder(
            self.expected_image_size, self.output_shape, self.N_CLASSES
        )

    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
        current_image_size: Optional[mnn.vision.image_size.ImageSize] = None,
    ) -> torch.Tensor:
        output_tensor_bboxes = torch.zeros(
            (
                self.output_shape.height,
                self.output_shape.width,
            )
        )

        for i, annotation in enumerate(annotations):
            # Check Category
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            # Translate bbox to padded image
            x1_norm, y1_norm, w_norm, h_norm = (
                mnn_bbox_mapper.translate_norm_bbox_to_padded_image(
                    annotation["normalized_bbox"],
                    fixed_ratio_components,
                    padding_percent,
                    self.expected_image_size,
                )
            )

            # Check bbox
            xc_norm, yc_norm, w_norm, h_norm = mnn_bbox_mapper.tl_xywh_to_center_xywh(
                (x1_norm, y1_norm, w_norm, h_norm)
            )

            if any(x < 0 or x > 1 for x in (xc_norm, yc_norm, w_norm, h_norm)):
                continue

            # Encode bbox center into grid
            (xc_norm_in_grid, yc_norm_in_grid), (position_x, position_y) = (
                self.grid_encoder_decoder.encode(xc_norm, yc_norm)
            )
            # Place vector in tensor/mask
            position_in_tensor = mnn_grid.twoD_grid_position_to_oneD_position(
                position_x,
                position_y,
                grid_S=self.grid_encoder_decoder.output_mask_grid_S,
            )

            new_bbox_norm = [
                xc_norm_in_grid,
                yc_norm_in_grid,
                w_norm,
                h_norm,
            ]

            # Create vector
            vector = self._create_object_vector(
                new_bbox_norm, category, self.output_shape.width
            )

            output_tensor_bboxes[position_in_tensor, :] = vector
            # for i in range(1, self.ORDINAL_EXPANSION + 1):
            #     if (
            #         position_in_tensor - i < 0
            #         or position_in_tensor + i >= self.output_shape.height
            #     ):
            #         continue
            #     if (
            #         output_tensor_bboxes[
            #             position_in_tensor - i, self.vector_indices.objectness_idx
            #         ]
            #         == 0
            #     ):
            #         output_tensor_bboxes[
            #             position_in_tensor - i, self.vector_indices.objectness_idx
            #         ] = (1 - i / self.ORDINAL_EXPANSION)
            #     if (
            #         output_tensor_bboxes[
            #             position_in_tensor + i, self.vector_indices.objectness_idx
            #         ]
            #         == 0
            #     ):
            #         output_tensor_bboxes[
            #             position_in_tensor + i, self.vector_indices.objectness_idx
            #         ] = (1 - i / self.ORDINAL_EXPANSION)
        return output_tensor_bboxes

    ##########################################################################################

    def output_to_class_scores_batch(self, y: torch.Tensor):
        return y[:, :, -len(self.classes) :]

    def split_output_to_vectors_batch(self, y: torch.Tensor):
        return self.vector_indices.split_mask_batch(y)

    def split_output_to_vectors(self, y: torch.Tensor):
        return self.vector_indices.split_mask(y)

    def _decode_bbox_raw_vector(
        self,
        xc_ordinals: torch.Tensor,
        yc_ordinals: torch.Tensor,
        w_ordinals: torch.Tensor,
        h_ordinals: torch.Tensor,
        vector_position: int,
    ):

        xc_norm_in_grid_cell = self._decode_coordinate_vector_norm(xc_ordinals)
        yc_norm_in_grid_cell = self._decode_coordinate_vector_norm(yc_ordinals)
        w_norm = self._decode_coordinate_vector_norm(
            w_ordinals,
        )
        h_norm = self._decode_coordinate_vector_norm(
            h_ordinals,
        )

        ###
        xc_norm, yc_norm = self.grid_encoder_decoder.decode(
            vector_position, xc_norm_in_grid_cell, yc_norm_in_grid_cell
        )

        # de-normalize based on shape of the expected image
        xc = xc_norm * self.expected_image_size.width
        yc = yc_norm * self.expected_image_size.height
        w0 = w_norm * self.expected_image_size.width
        h0 = h_norm * self.expected_image_size.height

        bbox = [xc, yc, w0, h0]
        return bbox

    OBJECTNESS_THRESHOLD = 0.1

    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        objects = y
        bboxes = []
        categories = []
        category_scores = []
        objectness_scores = []

        for i in range(objects.shape[0]):
            o = objects[i, :]

            (
                xc_ordinals,
                yc_ordinals,
                w_ordinals,
                h_ordinals,
                objectness_score,
                category_vector,
            ) = self.vector_indices.split_vector(o)

            if category_vector.sum() == 0:
                continue
            if objectness_score == 0:
                continue

            if (
                filter_by_objectness_score
                and objectness_score < self.OBJECTNESS_THRESHOLD
            ):
                continue
            # Bbox
            bbox = self._decode_bbox_raw_vector(
                xc_ordinals, yc_ordinals, w_ordinals, h_ordinals, i
            )
            bboxes.append(bbox)

            # Category
            category = torch.argmax(category_vector)
            category_score = category_vector[category]
            categories.append(category.item())
            category_scores.append(category_score.item())

            # Objectness
            objectness_scores.append(objectness_score.item())
        return (
            torch.Tensor(bboxes),
            torch.Tensor(categories),
            torch.Tensor(category_scores),
            torch.Tensor(objectness_scores),
        )

    def write_image_with_model_output(
        self, model_output: torch.Tensor, image: torch.Tensor, sub_dir: str
    ):
        bboxes, categories, categories_scores, objectness_scores = (
            self.decode_output_tensor(model_output, filter_by_objectness_score=True)
        )
        if "pred" in sub_dir:
            if bboxes.shape[0] > 20:
                bboxes = bboxes[:20]
                categories = categories[:20]
                categories_scores = categories_scores[:20]
                objectness_scores = objectness_scores[:20]

        validation_img = image.detach().cpu()
        validation_img = validation_img.permute(1, 2, 0)
        image = (validation_img.numpy() * 255).astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for bbox, category, category_score, objectness_score in zip(
            bboxes, categories, categories_scores, objectness_scores
        ):
            bbox_tlwh = mnn_bbox_mapper.center_xywh_to_tl_xywh_tensor(bbox)

            x1 = int(bbox_tlwh[0].item())
            y1 = int(bbox_tlwh[1].item())
            w = int(bbox_tlwh[2].item())
            h = int(bbox_tlwh[3].item())
            x2 = x1 + w
            y2 = y1 + h
            point1 = (x1, y1)
            point2 = (x2, y2)
            cv2.rectangle(image, point1, point2, (0, 255, 0), 2)

            category_no = int(category.item())
            confidence = objectness_score.item() * category_score.item()
            if confidence >= 1.0:
                cat = f"{category_no}"
            else:
                cat = f"{category_no}-{confidence:.3f}"
            cv2.putText(
                image,
                cat,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        msg = f"Objectness Threshold: {self.OBJECTNESS_THRESHOLD}"
        cv2.putText(
            image,
            msg,
            (40, image.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (240, 20, 0),
            1,
            cv2.LINE_AA,
        )

        # reverse mask
        os.makedirs(f"assessment_images/{sub_dir}", exist_ok=True)
        cv2.imwrite(f"assessment_images/{sub_dir}/bboxed_image.jpg", image)


class COCOInstances2017OrdinalBalanced(COCOInstances2017Ordinal):

    def get_pair(self, idx: int) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        image_tensor, annotations = super().get_pair(idx)
        if len(annotations) == 0:
            return image_tensor, annotations

        method = random.choice(["original", "object_crop"])
        if method == "object_crop":
            # Sort annotations by category
            annotations = sorted(annotations, key=lambda x: x["category_id"])

            class_ids_weights = [
                get_balanced_category_weight(annotation["category_id"])
                for annotation in annotations
            ]

            annotation = random.choices(annotations, weights=class_ids_weights)[0]
            x1, y1, w, h = annotation["normalized_bbox"]
            x1 = int(x1 * image_tensor.shape[2])
            y1 = int(y1 * image_tensor.shape[1])
            w = int(w * image_tensor.shape[2])
            h = int(h * image_tensor.shape[1])
            object_crop = image_tensor[:, y1 : y1 + h, x1 : x1 + w]
            new_image_tensor = torch.zeros_like(image_tensor)
            position_x1 = random.randint(0, new_image_tensor.shape[2] - w)
            position_y1 = random.randint(0, new_image_tensor.shape[1] - h)
            new_image_tensor[
                :, position_y1 : position_y1 + h, position_x1 : position_x1 + w
            ] = object_crop
            category_id = (
                annotation["category_id"] if annotation["category_id"] <= 78 else 78
            )  # I don't care about classes above 78. Either way their occurences are too low. I just need the object
            annotations = [
                {
                    "normalized_bbox": [
                        position_x1 / new_image_tensor.shape[2],
                        position_y1 / new_image_tensor.shape[1],
                        w / new_image_tensor.shape[2],
                        h / new_image_tensor.shape[1],
                    ],
                    "category_id": category_id,
                }
            ]
            return new_image_tensor, annotations
        return image_tensor, annotations
