"""
Consider the model's output is a tensor of shape (480, 640).
I will face it as an output of 480 vectors.
The first vector is the number of objects in the image.

The rest 479 vectors are the predictions for each object.
Each vector has 640 elements and we divide it into 6 parts:
1. objectness score
2. class as in multi-class classification. So a vector of 80 elements for COCO.
Actually I will use 79 classes and fuck the 80th class. I don't care. So we have 560 remaining elements
3. 560/4 = 140. I will use the vector with 140 to predict x1
4. 560/4 = 140. I will use the vector with 140 to predict y1
5. 560/4 = 140. I will use the vector with 140 to predict x2
6. 560/4 = 140. I will use the vector with 140 to predict y2
"""

import abc
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import cv2

from mnn.vision.dataset.coco.torch_dataset_csv import BaseCOCODatasetGroupedCsv
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.image_size

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
    N_CLASSES = 79

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        expected_image_size: mnn.vision.image_size.ImageSize,
    ):
        # 79 classes
        self.classes = [i for i in range(self.N_CLASSES)]
        super().__init__(data_dir, split, expected_image_size, self.classes)

        self.bbox_vector_size = (
            expected_image_size.width - len(self.classes) - 1
        )  # -1 for the objectness score
        if self.bbox_vector_size % 4 != 0:
            raise ValueError(
                "The number of elements in the vector for the bounding box must be divisible by 4. Make sure you (input_image.width - 80) is divisible by 4."
            )
        """
        NOTE: the model will learn to detect up to a certain number of objects and not more than that
        """
        self.max_objects = expected_image_size.width - 1
        # Why -1? Because the first vector is the number of objects in the image
        # The rest of the vectors are used to predict the objects

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

    def _create_number_of_objects_vector(
        self, n_objects: int, vector_size: int
    ) -> torch.Tensor:
        vector = torch.zeros(vector_size)

        # Left side
        vector = self._expand_left(vector, n_objects)

        # Middle
        vector[n_objects] = 1

        # Right side
        vector = self._expand_right(vector, n_objects)

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

        vector[0:coordinate_span_of_indices_length] = x1
        vector[
            coordinate_span_of_indices_length : 2 * coordinate_span_of_indices_length
        ] = y1
        vector[
            2
            * coordinate_span_of_indices_length : 3
            * coordinate_span_of_indices_length
        ] = x2
        vector[
            3
            * coordinate_span_of_indices_length : 4
            * coordinate_span_of_indices_length
        ] = y2

        # Category
        vector[self.bbox_vector_size + category] = 1
        # Objectness score
        vector[self.bbox_vector_size + len(self.classes)] = 1
        return vector

    def _decode_coordinate_vector(
        self, vector: torch.Tensor, image_dimension_size: int
    ) -> int:
        vector_size = vector.shape[0]
        idx = torch.argmax(vector).item()
        # idx = idx + 1 if idx < vector_size - 1 else idx
        normalized_coordinate = idx / (vector_size - 1)
        return int(normalized_coordinate * image_dimension_size)

    @abc.abstractmethod
    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        pass

    def map_bbox_to_padded_image(
        self,
        x1: float,  # TODO - there's a bug in the preparation of CSVs and the x1, y1 are probably xc, yc
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


class COCOInstances2017Ordinal(BaseCOCOInstances2017Ordinal):

    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
        current_image_size: Optional[mnn.vision.image_size.ImageSize] = None,
    ) -> torch.Tensor:
        output_tensor = torch.zeros(
            (self.expected_image_size.height, self.expected_image_size.width)
        )

        # Add vector with number of objects
        n_objects = len(output_tensor)
        if n_objects > self.max_objects:
            """Keep the 'max_objects' objects with the highest area"""
            # This point will never be reached, unless you're a dork and use an image width of less than 420 pixels.
            raise NotImplementedError(
                "Keeping the 'max_objects' objects with the highest area is not implemented yet"
            )
        output_tensor[0, :] = self._create_number_of_objects_vector(
            n_objects, self.expected_image_size.width
        )

        # Add whole image as a bounding box
        vectors_for_output = []
        for i, annotation in enumerate(annotations):
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]
            area = w_norm * h_norm
            # Skip very small bboxes. Bad annotations
            if area < 0.0004:
                continue
            # Skip very close to image borders bboxes. Bad annotations
            if (
                x1_norm > 0.99
                or y1_norm > 0.99
                or (x1_norm + w_norm) <= 0.01
                or (y1_norm + h_norm) <= 0.01
            ):
                continue

            x1 = x1_norm * fixed_ratio_components.resize_width
            y1 = y1_norm * fixed_ratio_components.resize_height
            w = w_norm * fixed_ratio_components.resize_width
            h = h_norm * fixed_ratio_components.resize_height
            x1, y1, w, h = self.map_bbox_to_padded_image(
                x1, y1, w, h, fixed_ratio_components, padding_percent
            )
            xc = x1 + w / 2
            yc = y1 + h / 2
            new_bbox_norm = [
                xc / self.expected_image_size.width,
                yc / self.expected_image_size.height,
                w / self.expected_image_size.width,
                h / self.expected_image_size.height,
            ]
            vector = self._create_object_vector(
                new_bbox_norm, category, self.expected_image_size.width
            )

            vectors_for_output.append((vector, area))

        # Sort by area and then place in output tensor, so there's at least a logic/order in the predictions
        sorted_vectors_for_output = sorted(vectors_for_output, key=lambda x: x[1])
        for i, (vector, area) in enumerate(sorted_vectors_for_output):
            output_tensor[1 + i, :] = vector
        return output_tensor

    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        n_vectors, vector_size = y.shape
        n_objects_vector = y[0, :]
        n_objects = torch.argmax(n_objects_vector)
        if n_objects > self.TOTAL_POSSIBLE_OBJECTS:
            # Unwanted behaviour by the model. Don't do anything. Keep it in mind.
            pass

        bbox_vector_size = vector_size - (self.N_CLASSES + 1)
        _coord_step = bbox_vector_size // 4

        objects = y[
            1 : self.TOTAL_POSSIBLE_OBJECTS + 1, :
        ]  # The rest should be ignored
        img_h, img_w = self.expected_image_size.height, self.expected_image_size.width

        bboxes = []
        categories = []
        objectness_scores = []
        for o in objects:
            objectness_score = o[-1]
            if filter_by_objectness_score and objectness_score < 0.5:
                continue
            bbox_raw = o[: (len(o) - 80)]
            xc = self._decode_coordinate_vector(bbox_raw[:_coord_step], img_w)
            yc = self._decode_coordinate_vector(
                bbox_raw[_coord_step : 2 * _coord_step], img_h
            )
            w = self._decode_coordinate_vector(
                bbox_raw[2 * _coord_step : 3 * _coord_step], img_w
            )
            h = self._decode_coordinate_vector(
                bbox_raw[3 * _coord_step : 4 * _coord_step], img_h
            )

            bbox = [xc, yc, w, h]
            if all(x == 0 for x in bbox):
                continue
            bboxes.append(bbox)
            category = torch.argmax(o[(len(o) - 80) : -1])
            categories.append(category)
            objectness_scores.append(objectness_score)

        # # Sort by objectness score
        # sorted_indices = sorted(
        #     range(len(objectness_scores)),
        #     key=lambda k: objectness_scores[k],
        #     reverse=True,
        # )
        # bboxes = [bboxes[i] for i in sorted_indices]
        # categories = [categories[i] for i in sorted_indices]
        # objectness_scores = [objectness_scores[i] for i in sorted_indices]
        return bboxes[:n_objects], categories[:n_objects], objectness_scores[:n_objects]


class COCOInstances2017Ordinal2(BaseCOCOInstances2017Ordinal):

    def _make_annotations_to_vectors(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float,
    ):
        vectors_for_output = []
        for i, annotation in enumerate(annotations):
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]
            area = w_norm * h_norm
            # Skip very small bboxes. Bad annotations
            if area < 0.0004:
                continue
            # Skip very close to image borders bboxes. Bad annotations
            if (
                x1_norm > 0.99
                or y1_norm > 0.99
                or (x1_norm + w_norm) <= 0.01
                or (y1_norm + h_norm) <= 0.01
            ):
                continue

            x1 = x1_norm * fixed_ratio_components.resize_width
            y1 = y1_norm * fixed_ratio_components.resize_height
            w = w_norm * fixed_ratio_components.resize_width
            h = h_norm * fixed_ratio_components.resize_height
            x1, y1, w, h = self.map_bbox_to_padded_image(
                x1, y1, w, h, fixed_ratio_components, padding_percent
            )
            xc = x1 + w / 2
            yc = y1 + h / 2

            new_bbox_norm = [
                xc / self.expected_image_size.width,
                yc / self.expected_image_size.height,
                w / self.expected_image_size.width,
                h / self.expected_image_size.height,
            ]

            vector = self._create_object_vector(
                new_bbox_norm, category, self.expected_image_size.width
            )

            vectors_for_output.append((vector, area))
        return vectors_for_output

    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
        current_image_size: Optional[mnn.vision.image_size.ImageSize] = None,
    ) -> torch.Tensor:
        output_tensor_bboxes = torch.zeros(
            (self.expected_image_size.height, self.expected_image_size.width)
        )

        # Add vector with number of objects
        n_objects = len(annotations)
        if n_objects > self.max_objects:
            """Keep the 'max_objects' objects with the highest area"""
            # This point will never be reached, unless you're a dork and use an image width of less than 420 pixels.
            raise NotImplementedError(
                "Keeping the 'max_objects' objects with the highest area is not implemented yet"
            )

        output_tensor_n_objects = torch.zeros(
            (self.expected_image_size.height, self.expected_image_size.width)
        )
        output_tensor_n_objects[0, :] = self._create_number_of_objects_vector(
            n_objects, self.expected_image_size.width
        )

        # Add whole image as a bounding box
        vectors_for_output = self._make_annotations_to_vectors(
            annotations, fixed_ratio_components, padding_percent
        )

        # FAIL
        # Sort by area and then place in output tensor, so there's at least a logic/order in the predictions
        sorted_vectors_for_output = sorted(vectors_for_output, key=lambda x: x[1])
        for i, (vector, area) in enumerate(sorted_vectors_for_output):
            output_tensor_bboxes[i, :] = vector
        return torch.stack([output_tensor_n_objects, output_tensor_bboxes])

    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        """
        y: torch.Tensor --> Stack [output_tensor_n_objects, output_tensor_bboxes]
        """
        # Separate the two tensors
        n_objects_vector = y[0, 0, :]
        objects = y[1, : self.TOTAL_POSSIBLE_OBJECTS, :]

        # Extract number of objects
        n_objects = min(torch.argmax(n_objects_vector), self.TOTAL_POSSIBLE_OBJECTS)
        if n_objects == 0:
            return [], [], []

        # Extract bboxes
        _coord_step = self.bbox_vector_size // 4
        h, w = self.expected_image_size.height, self.expected_image_size.width
        bboxes = []
        categories = []
        objectness_scores = []
        for i, o in enumerate(objects):
            objectness_score = o[-1]
            if filter_by_objectness_score and objectness_score < 0.5:
                continue
            total_classes = len(self.classes)
            vector_size = len(o)
            idx_bbox = vector_size - (total_classes + 1)
            bbox_raw = o[:idx_bbox]
            xc = self._decode_coordinate_vector(bbox_raw[:_coord_step], w)
            yc = self._decode_coordinate_vector(
                bbox_raw[_coord_step : 2 * _coord_step], h
            )
            w0 = self._decode_coordinate_vector(
                bbox_raw[2 * _coord_step : 3 * _coord_step], w
            )
            h0 = self._decode_coordinate_vector(
                bbox_raw[3 * _coord_step : 4 * _coord_step], h
            )

            if all(x == 0 for x in [xc, yc, w0, h0]):
                continue

            bbox = [xc, yc, w0, h0]
            idx_category = idx_bbox + total_classes
            category = torch.argmax(o[idx_bbox:idx_category])
            bboxes.append(bbox)
            categories.append(category)
            objectness_scores.append(objectness_score)

        return bboxes[:n_objects], categories[:n_objects], objectness_scores[:n_objects]

    def decode_prediction_raw(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        predicted_objects = y[1, :, :]
        # Extract bboxes
        _coord_step = self.bbox_vector_size // 4
        h, w = self.expected_image_size.height, self.expected_image_size.width
        bboxes = []
        categories = []
        objectness_scores = []
        for i, o in enumerate(predicted_objects):
            objectness_score = o[-1]
            if filter_by_objectness_score and objectness_score < 0.5:
                continue
            total_classes = len(self.classes)
            vector_size = len(o)
            idx_bbox = vector_size - (total_classes + 1)
            bbox_raw = o[:idx_bbox]
            xc = self._decode_coordinate_vector(bbox_raw[:_coord_step], w)
            yc = self._decode_coordinate_vector(
                bbox_raw[_coord_step : 2 * _coord_step], h
            )
            w0 = self._decode_coordinate_vector(
                bbox_raw[2 * _coord_step : 3 * _coord_step], w
            )
            h0 = self._decode_coordinate_vector(
                bbox_raw[3 * _coord_step : 4 * _coord_step], h
            )

            if all(x == 0 for x in [xc, yc, w0, h0]):
                continue

            bbox = [xc, yc, w0, h0]
            idx_category = idx_bbox + total_classes
            category = torch.argmax(o[idx_bbox:idx_category])
            bboxes.append(bbox)
            categories.append(category)
            objectness_scores.append(objectness_score)

        return bboxes, categories, objectness_scores


class COCOInstances2017Ordinal3(BaseCOCOInstances2017Ordinal):

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        expected_image_size: mnn.vision.image_size.ImageSize,
    ):
        super().__init__(data_dir, split, expected_image_size)
        if self.expected_image_size.height ** (1 / 2) % 1 != 0:
            raise ValueError(
                f"The square root of the height of the expected image size for '{__class__}' should be an integer."
            )

        self.grid_S = int(self.expected_image_size.height ** (1 / 2))

    def _calculate_position_in_grid(
        self, xc_norm: float, yc_norm: float
    ) -> Tuple[int, int]:
        # TODO - is it possible to run in O(1) time?

        step = 1 / self.grid_S
        for i in range(self.grid_S):
            if i * step <= xc_norm <= (i + 1) * step:
                x = i
            if i * step <= yc_norm <= (i + 1) * step:
                y = i
        return x, y

    def _calculate_coordinate_in_grid(self, coord_norm: float, position: int) -> float:
        step = 1 / self.grid_S
        lower_bound = position * step
        return (coord_norm - lower_bound) / step

    def _calculate_position_in_tensor_from_grid(self, x: int, y: int) -> int:
        return y * self.grid_S + x

    def _calculate_position_in_grid_from_tensor(self, position: int) -> Tuple[int, int]:
        y = position // self.grid_S
        x = position % self.grid_S
        return x, y

    def _make_annotations_to_vectors_and_place_in_output_tensor(
        self,
        dst_tensor: torch.Tensor,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float,
    ):
        for i, annotation in enumerate(annotations):
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]

            area = w_norm * h_norm
            # Skip very small bboxes. Bad annotations
            if area < 0.0004:
                continue
            # Skip very close to image borders bboxes. Bad annotations
            if (
                x1_norm > 0.99
                or y1_norm > 0.99
                or (x1_norm + w_norm) <= 0.01
                or (y1_norm + h_norm) <= 0.01
            ):
                continue

            vector, position_x, position_y = self._transform_annotation_into_vector(
                annotation, fixed_ratio_components, padding_percent
            )
            position_in_tensor = self._calculate_position_in_tensor_from_grid(
                position_x, position_y
            )
            # NOTE - objects can be replaced in case they belong to the same grid cell
            dst_tensor[position_in_tensor, :] = vector
        return dst_tensor

    def _transform_annotation_into_vector(
        self,
        annotation: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float,
    ) -> torch.Tensor:
        x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]

        x1 = x1_norm * fixed_ratio_components.resize_width
        y1 = y1_norm * fixed_ratio_components.resize_height
        w = w_norm * fixed_ratio_components.resize_width
        h = h_norm * fixed_ratio_components.resize_height
        x1, y1, w, h = self.map_bbox_to_padded_image(
            x1, y1, w, h, fixed_ratio_components, padding_percent
        )
        xc = x1 + w / 2
        yc = y1 + h / 2

        new_bbox_norm = [
            xc / self.expected_image_size.width,
            yc / self.expected_image_size.height,
            w / self.expected_image_size.width,
            h / self.expected_image_size.height,
        ]
        xc_norm = new_bbox_norm[0]
        yc_norm = new_bbox_norm[1]
        position_x, position_y = self._calculate_position_in_grid(xc_norm, yc_norm)
        xc_in_grid_norm = self._calculate_coordinate_in_grid(xc_norm, position_x)
        yc_in_grid_norm = self._calculate_coordinate_in_grid(yc_norm, position_y)
        new_bbox_norm[0] = xc_in_grid_norm
        new_bbox_norm[1] = yc_in_grid_norm

        category = int(annotation["category_id"]) - 1
        vector = self._create_object_vector(
            new_bbox_norm, category, self.expected_image_size.width
        )
        return vector, position_x, position_y

    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
        current_image_size: Optional[mnn.vision.image_size.ImageSize] = None,
    ) -> torch.Tensor:
        output_tensor_bboxes = torch.zeros(
            (self.expected_image_size.height, self.expected_image_size.width)
        )

        for i, annotation in enumerate(annotations):
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]

            area = w_norm * h_norm
            # Skip very small bboxes. Bad annotations
            if area < 0.0004:
                continue
            # Skip very close to image borders bboxes. Bad annotations
            if (
                x1_norm > 0.99
                or y1_norm > 0.99
                or (x1_norm + w_norm) <= 0.01
                or (y1_norm + h_norm) <= 0.01
            ):
                continue

            vector, position_x, position_y = self._transform_annotation_into_vector(
                annotation, fixed_ratio_components, padding_percent
            )
            position_in_tensor = (
                position_y * self.grid_S + position_x
            )  # in 'height' dimension
            output_tensor_bboxes[position_in_tensor, :] = vector

        return output_tensor_bboxes

    def _decode_coordinate_vector_norm(self, vector: torch.Tensor) -> int:
        vector_size = vector.shape[0]
        idx = torch.argmax(vector).item()
        return idx / (vector_size - 1)

    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        objects = y
        _coord_step = self.bbox_vector_size // 4
        h, w = self.expected_image_size.height, self.expected_image_size.width
        bboxes = []
        categories = []
        objectness_scores = []
        for i, o in enumerate(objects):
            objectness_score = o[-1]
            total_classes = len(self.classes)
            vector_size = len(o)
            idx_bbox = vector_size - (total_classes + 1)
            bbox_raw = o[:idx_bbox]
            xc_norm_in_grid_cell = self._decode_coordinate_vector_norm(
                bbox_raw[:_coord_step]
            )
            yc_norm_in_grid_cell = self._decode_coordinate_vector_norm(
                bbox_raw[_coord_step : 2 * _coord_step]
            )
            w0 = self._decode_coordinate_vector(
                bbox_raw[2 * _coord_step : 3 * _coord_step], w
            )
            h0 = self._decode_coordinate_vector(
                bbox_raw[3 * _coord_step : 4 * _coord_step], h
            )

            Sx, Sy = self._calculate_position_in_grid_from_tensor(i)
            step_x_in_pixels = w // self.grid_S
            step_y_in_pixels = h // self.grid_S
            xc = int((Sx + xc_norm_in_grid_cell) * step_x_in_pixels)
            yc = int((Sy + yc_norm_in_grid_cell) * step_y_in_pixels)

            if all(x == 0 for x in [xc, yc, w0, h0]):
                continue

            bbox = [xc, yc, w0, h0]
            idx_category = idx_bbox + total_classes
            category = torch.argmax(o[idx_bbox:idx_category])
            bboxes.append(bbox)
            categories.append(category)
            objectness_scores.append(objectness_score.item())

        return bboxes, categories, objectness_scores

    def filter_low_objectness(
        self, y: torch.Tensor, objectness_threshold: float
    ) -> torch.Tensor:
        for i in range(y.shape[0]):
            obj = y[i]
            objectness_score = obj[-1]
            if objectness_score < objectness_threshold:
                y[i] = torch.zeros_like(obj)
        return y

    def write_image_with_model_output(
        self, model_output: torch.Tensor, image: torch.Tensor, sub_dir: str
    ):
        model_output = self.filter_low_objectness(model_output, 0.75)
        bboxes, categories, objectness_scores = self.decode_output_tensor(model_output)

        validation_img = image.detach().cpu()
        validation_img = validation_img.permute(1, 2, 0)
        image = (validation_img.numpy() * 255).astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for bbox, category, confidence in zip(bboxes, categories, objectness_scores):
            if confidence <= 0.001:
                continue

            xc, yc, w, h = bbox
            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = x1 + int(w)
            y2 = y1 + int(h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

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


def write_image_with_output(
    temp_out: torch.Tensor,
    validation_image: torch.Tensor,
    sub_dir: str = "any",
):
    bboxes, categories, objectness_scores = (
        COCOInstances2017Ordinal.decode_output_tensor(temp_out.squeeze(0))
    )

    validation_img = validation_image.squeeze(0).detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    image = (validation_img.numpy() * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for bbox, category in zip(bboxes, categories):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(category.item()),
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


if __name__ == "__main__":
    import pathlib
    import mnn.vision.image_size
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()
    idx = args.idx

    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
    )
    expected_image_size = mnn.vision.image_size.ImageSize(width=640, height=480)

    val_dataset = COCOInstances2017Ordinal(dataset_dir, "val", expected_image_size)
    image_batch, target0 = val_dataset[idx]
    write_image_with_output(
        target0.unsqueeze(0),
        image_batch.unsqueeze(0),
        "train_image_ground_truth",
    )
    image, target = val_dataset.get_pair(idx)
    image = image.permute(1, 2, 0)
    image = (image.numpy() * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for annotation in target:
        bbox = [int(x) for x in annotation["bbox"]]
        cv2.rectangle(
            image,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (0, 255, 0),
            2,
        )
    cv2.imwrite("assessment_images/train_image_ground_truth/ground_truth.jpg", image)
