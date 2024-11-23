import abc
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import cv2

from mnn.vision.dataset.coco.torch_dataset_csv import BaseCOCODatasetGroupedCsv
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.image_size
import mnn.vision.process_output.object_detection.bbox_mapper as mnn_bbox_mapper
import mnn.vision.process_output.object_detection.grid as mnn_grid

import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)


class COCODatasetInstances2017(BaseCOCODatasetGroupedCsv):

    def get_year(self) -> int:
        return "2017"

    def get_type(self) -> str:
        return "instances"


class BaseCOCOInstances2017Ordinal(COCODatasetInstances2017):
    """
    Everything Ordinal
    """

    ORDINAL_EXPANSION = 4
    TOTAL_POSSIBLE_OBJECTS = 98  # COCO has max 98 objects in one image
    N_CLASSES = 80

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        expected_image_size: mnn.vision.image_size.ImageSize,
        output_shape: mnn.vision.image_size.ImageSize = None,
    ):
        if output_shape is None:
            self.output_shape = expected_image_size
        else:
            self.output_shape = output_shape

        self.classes = [i for i in range(self.N_CLASSES)]
        super().__init__(data_dir, split, expected_image_size, self.classes)

        self.vector_size = self.output_shape.width
        self.classes_vector_size = len(self.classes)
        self.bbox_vector_size = self.vector_size - self.classes_vector_size
        if self.bbox_vector_size % 4 != 0:
            raise ValueError(
                "The number of elements in the vector for the bounding box must be divisible by 4. Make sure you (input_image.width - 80) is divisible by 4."
            )

        self.max_objects = self.output_shape.width

    @abc.abstractmethod
    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        pass

    #####################################################################################

    def _expand_left(self, vector: torch.Tensor, idx: int):
        left_side = idx - self.ORDINAL_EXPANSION
        if left_side < 0:
            left_side = 0
        d = idx - left_side
        for i in range(1, d):
            probability = i / self.ORDINAL_EXPANSION
            vector[idx - (d - i)] = probability

        return vector

    def _expand_right(self, vector: torch.Tensor, idx: int):
        vector_size = len(vector)
        # Right side
        right_side = idx + self.ORDINAL_EXPANSION
        if right_side > vector_size:
            right_side = vector_size
        d = right_side - idx
        for i in range(1, d):
            probability = i / self.ORDINAL_EXPANSION
            vector[idx + (d - i)] = probability
        return vector

    def _create_object_vector(self, bbox: list[float], category: int, vector_size: int):
        """expecting bbox as x1, y1, x2, y2"""
        vector = torch.zeros(vector_size)

        x1_coord_norm, y1_coord_norm, x2_coord_norm, y2_coord_norm = bbox
        # Bbox
        coordinate_span_of_indices_length = self.bbox_vector_size // 4
        scale_factor = coordinate_span_of_indices_length - 1

        idx = round(x1_coord_norm * scale_factor)
        x1 = torch.zeros(coordinate_span_of_indices_length)
        x1[idx] = 1
        x1 = self._expand_left(x1, idx)
        x1 = self._expand_right(x1, idx)

        idx = round(y1_coord_norm * scale_factor)
        y1 = torch.zeros(coordinate_span_of_indices_length)
        y1[idx] = 1
        y1 = self._expand_left(y1, idx)
        y1 = self._expand_right(y1, idx)

        idx = round(x2_coord_norm * scale_factor)
        x2 = torch.zeros(coordinate_span_of_indices_length)
        x2[idx] = 1
        x2 = self._expand_left(x2, idx)
        x2 = self._expand_right(x2, idx)

        idx = round(y2_coord_norm * scale_factor)
        y2 = torch.zeros(coordinate_span_of_indices_length)
        y2[idx] = 1
        y2 = self._expand_left(y2, idx)
        y2 = self._expand_right(y2, idx)

        coord_len = coordinate_span_of_indices_length
        vector[:coord_len] = x1
        vector[coord_len : 2 * coord_len] = y1
        vector[2 * coord_len : 3 * coord_len] = x2
        vector[3 * coord_len : 4 * coord_len] = y2

        # Category
        idx = category
        category_vector = torch.zeros(self.classes_vector_size)
        category_vector[idx] = 1
        category_vector = self._expand_left(category_vector, idx)
        category_vector = self._expand_right(category_vector, idx)
        vector[-self.classes_vector_size :] = category_vector
        return vector

    def _decode_coordinate_vector(
        self, vector: torch.Tensor, image_dimension_size: int
    ) -> int:
        if len(vector.shape) != 1:
            raise ValueError("The vector should be 1-dimensional")
        vector_size = vector.shape[0]
        idx = torch.argmax(vector).item()
        normalized_coordinate = idx / (vector_size - 1)
        return int(normalized_coordinate * image_dimension_size)

    def _decode_coordinate_vector_norm(self, vector: torch.Tensor) -> int:
        if len(vector.shape) != 1:
            raise ValueError("The vector should be 1-dimensional")
        vector_size = vector.shape[0]
        idx = torch.argmax(vector).item()
        return idx / (vector_size - 1)

    def _decode_coordinate_vector_batch(
        self, vector: torch.Tensor, image_dimension_size: int
    ) -> torch.Tensor:
        vector_size = vector.shape[2]
        indices = torch.argmax(vector, dim=2)
        normalized_coordinate = indices / (vector_size - 1)
        return normalized_coordinate * image_dimension_size

    def _decode_coordinate_vector_norm_batch(self, vector: torch.Tensor) -> int:
        vector_size = vector.shape[2]
        indices = torch.argmax(vector, dim=2)
        return indices / (vector_size - 1)


class COCOInstances2017Ordinal(BaseCOCOInstances2017Ordinal):

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        expected_image_size: mnn.vision.image_size.ImageSize,
        output_shape: mnn.vision.image_size.ImageSize = None,
    ):
        super().__init__(
            data_dir, split, expected_image_size, output_shape=output_shape
        )
        if self.output_shape.height ** (1 / 2) % 1 != 0:
            raise ValueError(
                f"The square root of the height of the expected image size for '{__class__}' should be an integer."
            )

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

    def _transform_annotation_into_vector(
        self,
        annotation: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:

        x1_norm, y1_norm, w_norm, h_norm = (
            mnn_bbox_mapper.translate_norm_bbox_to_padded_image(
                annotation["normalized_bbox"],
                fixed_ratio_components,
                padding_percent,
                self.expected_image_size,
            )
        )

        ##
        xc_norm, yc_norm, w_norm, h_norm = mnn_bbox_mapper.tl_xywh_to_center_xywh(
            (x1_norm, y1_norm, w_norm, h_norm)
        )

        if any(x < 0 or x > 1 for x in (xc_norm, yc_norm, w_norm, h_norm)):
            return torch.empty(0), (-1, -1)

        position_x, position_y = mnn_grid.calculate_grid(
            xc_norm, yc_norm, self.output_shape, self.output_mask_grid_S
        )

        in_grid_x, in_grid_y = mnn_grid.calculate_coordinate_in_grid(
            xc_norm, yc_norm, self.output_shape, self.output_mask_grid_S
        )

        new_bbox_norm = [in_grid_x, in_grid_y, w_norm, h_norm]

        # 2 - prepare category
        category = int(annotation["category_id"]) - 1

        # create vector
        vector = self._create_object_vector(
            new_bbox_norm, category, self.output_shape.width
        )
        return vector, (position_x, position_y)

    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
        current_image_size: Optional[mnn.vision.image_size.ImageSize] = None,
    ) -> torch.Tensor:
        output_tensor_bboxes = torch.zeros(
            (self.output_shape.height, self.output_shape.width)
        )

        for i, annotation in enumerate(annotations):
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            vector, (position_x, position_y) = self._transform_annotation_into_vector(
                annotation, fixed_ratio_components, padding_percent
            )
            if position_x == -1 or position_y == -1:
                continue

            position_in_tensor = mnn_grid.twoD_grid_position_to_oneD_position(
                position_x, position_y, grid_S=self.output_mask_grid_S
            )
            output_tensor_bboxes[position_in_tensor, :] = vector

        return output_tensor_bboxes

    ##########################################################################################

    def output_to_class_scores_batch(self, y: torch.Tensor):
        return y[:, :, -len(self.classes) :]

    def split_output_to_vectors_batch(self, y: torch.Tensor):
        coord_len = self.bbox_vector_size // 4
        xc_ordinals = y[:, :, :coord_len]
        yc_ordinals = y[:, :, coord_len : 2 * coord_len]
        w_ordinals = y[:, :, 2 * coord_len : 3 * coord_len]
        h_ordinals = y[:, :, 3 * coord_len : 4 * coord_len]
        class_scores = y[:, :, -len(self.classes) :]
        return (
            xc_ordinals,
            yc_ordinals,
            w_ordinals,
            h_ordinals,
            class_scores,
        )

    def split_output_to_vectors(self, y: torch.Tensor):
        coord_len = self.bbox_vector_size // 4
        xc_ordinals = y[:, :coord_len]
        yc_ordinals = y[:, coord_len : 2 * coord_len]
        w_ordinals = y[:, 2 * coord_len : 3 * coord_len]
        h_ordinals = y[:, 3 * coord_len : 4 * coord_len]
        class_scores = y[:, -len(self.classes) :]
        return (
            xc_ordinals,
            yc_ordinals,
            w_ordinals,
            h_ordinals,
            class_scores,
        )

    def write_image_with_model_output(
        self, model_output: torch.Tensor, image: torch.Tensor, sub_dir: str
    ):
        bboxes, categories, categories_scores = self.decode_output_tensor(model_output)

        # conf_threshold = 0.5
        # while categories_scores[categories_scores > 0.5].shape[0] == 0:
        #     conf_threshold -= 0.005
        #     if conf_threshold <= 0:
        #         conf_threshold = 0
        #         break

        # bboxes = bboxes[categories_scores > conf_threshold]
        # categories = categories[categories_scores > conf_threshold]
        validation_img = image.detach().cpu()
        validation_img = validation_img.permute(1, 2, 0)
        image = (validation_img.numpy() * 255).astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for bbox, category, confidence in zip(bboxes, categories, categories_scores):
            xc, yc, w, h = bbox
            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            if any(x < 0 for x in [x1, y1, x2, y2]):
                continue
            print(
                "OUT |",
                "Bbox: ",
                x1,
                y1,
                w,
                h,
                "Category: ",
                category,
            )
            cv2.rectangle(image, (x1, x2), (x2, y2), (0, 255, 0), 2)

            category_no = category.item()
            cat = (
                f"{category_no} - {confidence:.3f}"
                if category_no < 1.0
                else f"{category_no}"
            )
            cv2.putText(
                image,
                cat,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # reverse mask
        os.makedirs(f"assessment_images/{sub_dir}", exist_ok=True)
        cv2.imwrite(f"assessment_images/{sub_dir}/bboxed_image.jpg", image)

    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        objects = y
        _coord_step = self.bbox_vector_size // 4
        bboxes = []
        categories = []
        confidence_scores = []

        objects_positions = []
        for i in range(y.shape[0]):
            o = objects[i]
            if o.sum() == 0:
                continue
            objects_positions.append(i)

            ## BBox
            bbox_raw = o[: self.bbox_vector_size]
            xc_norm_in_grid_cell = self._decode_coordinate_vector_norm(
                bbox_raw[:_coord_step]
            )
            yc_norm_in_grid_cell = self._decode_coordinate_vector_norm(
                bbox_raw[_coord_step : 2 * _coord_step]
            )
            w0 = self._decode_coordinate_vector(
                bbox_raw[2 * _coord_step : 3 * _coord_step],
                self.expected_image_size.width,
            )
            h0 = self._decode_coordinate_vector(
                bbox_raw[3 * _coord_step : 4 * _coord_step],
                self.expected_image_size.height,
            )

            Sx, Sy = mnn_grid.oneD_position_to_twoD_grid_position(
                i, self.output_mask_grid_S
            )
            xc_norm, yc_norm = mnn_grid.calculate_real_coordinate(
                xc_norm_in_grid_cell,
                yc_norm_in_grid_cell,
                Sx,
                Sy,
                self.image_grid_S,
                self.expected_image_size,
            )
            xc = xc_norm * self.expected_image_size.width
            yc = yc_norm * self.expected_image_size.height

            if all(x == 0 for x in [xc, yc, w0, h0]):
                continue

            bbox = [xc, yc, w0, h0]

            # Category
            category_vector = o[self.bbox_vector_size :]
            category = torch.argmax(category_vector)
            category_score = category_vector[category]
            bboxes.append(bbox)
            categories.append(category.item())
            confidence_scores.append(category_score.item())
        return (
            torch.Tensor(bboxes),
            torch.Tensor(categories),
            torch.Tensor(confidence_scores),
        )
