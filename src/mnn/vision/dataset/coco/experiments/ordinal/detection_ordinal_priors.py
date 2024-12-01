import os
import pathlib
from typing import Any, Dict, Optional, Tuple

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

        # YOLOv3 bbox priors: https://arxiv.org/pdf/1804.02767
        self.priors = [
            (10, 13),  # small & square
            (16, 30),  # small & wide
            (33, 23),  # small & tall
            (30, 61),  # medium & wide
            (62, 45),  # medium & tall
            (59, 119),  # medium & wide
            (116, 90),  # large & tall
            (156, 198),  # large & square-wide
            (373, 326),  # large & square-tall
        ]
        self.prior_encoder_decoders = [
            mnn_vector_encoder.PriorEncoderDecoder(
                prior_box_shape=mnn.vision.image_size.ImageSize(width=p[0], height=p[1])
            )
            for p in self.priors
        ]
        self.n_priors = len(self.priors)

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
                len(self.prior_encoder_decoders),
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

            # Encode bbox w-h into priors
            for prior_i, prior_enc_dec in enumerate(self.prior_encoder_decoders):
                new_w, new_h = prior_enc_dec.encode(
                    round(w_norm * self.output_shape.width),
                    round(h_norm * self.output_shape.height),
                )
                new_bbox_norm = [
                    xc_norm_in_grid,
                    yc_norm_in_grid,
                    new_w,
                    new_h,
                ]

                # Create vector
                vector = self._create_object_vector(
                    new_bbox_norm, category, self.output_shape.width
                )

                output_tensor_bboxes[position_in_tensor, prior_i, :] = vector
        return output_tensor_bboxes

    ##########################################################################################

    def output_to_class_scores_batch(self, y: torch.Tensor, prior_i: int):
        return y[:, :, prior_i, -len(self.classes) :]

    def split_output_to_vectors_batch(self, y: torch.Tensor, prior_i: int):
        return self.vector_indices.split_mask_batch(y[:, :, prior_i, :])

    def split_output_to_vectors(self, y: torch.Tensor, prior_i: int):
        return self.vector_indices.split_mask(y[:, prior_i, :])

    def _decode_bbox_raw_vector(
        self,
        xc_ordinals: torch.Tensor,
        yc_ordinals: torch.Tensor,
        w_ordinals: torch.Tensor,
        h_ordinals: torch.Tensor,
        vector_position: int,
        prior_i: int,
    ):

        xc_norm_in_grid_cell = self._decode_coordinate_vector_norm(xc_ordinals)
        yc_norm_in_grid_cell = self._decode_coordinate_vector_norm(yc_ordinals)
        w0 = self._decode_coordinate_vector(
            w_ordinals,
            self.expected_image_size.width,
        )
        h0 = self._decode_coordinate_vector(
            h_ordinals,
            self.expected_image_size.height,
        )
        xc_norm, yc_norm = self.grid_encoder_decoder.decode(
            vector_position, xc_norm_in_grid_cell, yc_norm_in_grid_cell
        )
        w, h = self.prior_encoder_decoders[prior_i].decode(w0, h0)
        w_norm = w / self.expected_image_size.width
        h_norm = h / self.expected_image_size.height
        # de-normalize based on shape of the expected image
        xc = xc_norm * self.expected_image_size.width
        yc = yc_norm * self.expected_image_size.height

        bbox = [xc, yc, w_norm, h_norm]
        return bbox

    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """WITHOUT PRIORS - to be used only for Ground Truth"""
        objects = y
        bboxes = []
        categories = []
        category_scores = []
        objectness_scores = []

        objects_positions = []
        prior = 6
        for i in range(y.shape[0]):
            o = objects[i, prior, :]  # Use prior 6 - (116, 90)

            if o.sum() == 0:
                continue
            objects_positions.append(i)

            (
                xc_ordinals,
                yc_ordinals,
                w_ordinals,
                h_ordinals,
                objectness_score,
                category_vector,
            ) = self.vector_indices.split_vector(o)
            if filter_by_objectness_score and objectness_score < 0.5:
                continue
            # Bbox
            bbox = self._decode_bbox_raw_vector(
                xc_ordinals, yc_ordinals, w_ordinals, h_ordinals, i, prior
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

    def decode_output_tensor_with_priors(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        objects = y
        bboxes = []
        categories = []
        category_scores = []
        objectness_scores = []

        # TODO - make it faster, removing loops
        for prior in range(y.shape[1]):
            prior_objects = objects[:, prior, :]
            prior_objects = prior_objects[prior_objects.sum(axis=1) != 0]
            (
                all_xc_ordinals,
                all_yc_ordinals,
                all_w_ordinals,
                all_h_ordinals,
                all_objectness_scores,
                all_class_scores,
            ) = self.vector_indices.split_mask(prior_objects)
            all_xc_ordinals = all_xc_ordinals[all_objectness_scores >= 0.5]
            all_yc_ordinals = all_yc_ordinals[all_objectness_scores >= 0.5]
            all_w_ordinals = all_w_ordinals[all_objectness_scores >= 0.5]
            all_h_ordinals = all_h_ordinals[all_objectness_scores >= 0.5]
            all_class_scores = all_class_scores[all_objectness_scores >= 0.5]
            all_objectness_scores = all_objectness_scores[all_objectness_scores >= 0.5]
            for i in range(all_xc_ordinals.shape[0]):
                xc_ordinals = all_xc_ordinals[i]
                yc_ordinals = all_yc_ordinals[i]
                w_ordinals = all_w_ordinals[i]
                h_ordinals = all_h_ordinals[i]
                objectness_score = all_objectness_scores[i]
                class_scores = all_class_scores[i]
                # Bbox
                bbox = self._decode_bbox_raw_vector(
                    xc_ordinals, yc_ordinals, w_ordinals, h_ordinals, i, prior
                )
                bboxes.append(bbox)

                # Category
                category = torch.argmax(class_scores)
                category_score = class_scores[category]
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
            self.decode_output_tensor(model_output)
        )

        threshold = 0.5
        while objectness_scores[objectness_scores > threshold].shape[0] == 0:
            threshold -= 0.01
            if threshold < 0:
                threshold = 1  # no objects found
                break

        indices = objectness_scores > threshold
        bboxes = bboxes[indices]
        categories = categories[indices]
        categories_scores = categories_scores[indices]
        objectness_scores = objectness_scores[indices]

        validation_img = image.detach().cpu()
        validation_img = validation_img.permute(1, 2, 0)
        image = (validation_img.numpy() * 255).astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for bbox, category, confidence in zip(bboxes, categories, categories_scores):
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
            if confidence >= 1.0:
                cat = f"{category_no}"
            else:
                cat = f"{category_no} - {confidence:.3f}"
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
        if threshold == 1:
            msg = "No objects found"
        else:
            msg = f"Threshold: {threshold:.2f}"
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
