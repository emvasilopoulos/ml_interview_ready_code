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

import os
import pathlib
from typing import Any, Dict, Optional

import torch
import cv2

from mnn.vision.dataset.coco.torch_dataset import COCODatasetInstances2017
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.process_output.object_detection.rectangles_to_mask as mnn_rectangles_to_mask
import mnn.vision.image_size


class COCOInstances2017Ordinal(COCODatasetInstances2017):
    """
    Everything Ordinal
    """

    ORDINAL_EXPANSION = 4

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        expected_image_size: mnn.vision.image_size.ImageSize,
    ):
        # 79 classes
        self.classes = [i for i in range(79)]
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
        x1 = torch.zeros(coordinate_span_of_indices_length)
        idx = round(x1_coord_norm * coordinate_span_of_indices_length)
        x1[idx] = 1
        x1 = self._expand_left(x1, idx)
        x1 = self._expand_right(x1, idx)

        idx = round(y1_coord_norm * coordinate_span_of_indices_length)
        y1 = torch.zeros(coordinate_span_of_indices_length)
        y1[idx] = 1
        y1 = self._expand_left(y1, idx)
        y1 = self._expand_right(y1, idx)

        idx = round(x2_coord_norm * coordinate_span_of_indices_length) - 1
        x2 = torch.zeros(coordinate_span_of_indices_length)
        x2[idx] = 1
        x2 = self._expand_left(x2, idx)
        x2 = self._expand_right(x2, idx)

        idx = round(y2_coord_norm * coordinate_span_of_indices_length) - 1
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
        for i, annotation in enumerate(annotations):
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            x1, y1, w, h = annotation["normalized_bbox"]

            if fixed_ratio_components.pad_dimension == 1:
                pad_amount = (
                    fixed_ratio_components.expected_dimension_size
                    - fixed_ratio_components.resize_height
                )
                top_pad = int(pad_amount * padding_percent)

                y1 = y1 * fixed_ratio_components.resize_height + top_pad
                h = h * fixed_ratio_components.resize_height
                if y1 + h > self.expected_image_size.height:
                    h = self.expected_image_size.height - y1

                x1 = x1 * fixed_ratio_components.resize_width
                w = w * fixed_ratio_components.resize_width
            elif fixed_ratio_components.pad_dimension == 2:
                pad_amount = (
                    fixed_ratio_components.expected_dimension_size
                    - fixed_ratio_components.resize_width
                )
                left_pad = int(pad_amount * padding_percent)

                x1 = x1 * fixed_ratio_components.resize_width + left_pad
                w = w * fixed_ratio_components.resize_width
                if x1 + w > self.expected_image_size.width:
                    w = self.expected_image_size.width - x1

                y1 = y1 * fixed_ratio_components.resize_height
                h = h * fixed_ratio_components.resize_height
            else:
                raise ValueError("The pad_dimension should be 1 or 2")

            x1 = int(x1)
            y1 = int(y1)
            w = int(w)
            h = int(h)

            x2 = (
                x1 + w
                if x1 + w < self.expected_image_size.width
                else self.expected_image_size.width
            )
            y2 = (
                y1 + h
                if y1 + h < self.expected_image_size.height
                else self.expected_image_size.height
            )
            bbox_ = [
                x1 / self.expected_image_size.width,
                y1 / self.expected_image_size.height,
                x2 / self.expected_image_size.width,
                y2 / self.expected_image_size.height,
            ]
            vector = self._create_object_vector(
                bbox_, category, self.expected_image_size.width
            )

            output_tensor[i + 1, :] = vector
        return output_tensor


def _decode_coordinate_vector(vector: torch.Tensor, image_dimension_size: int) -> int:
    vector_size = vector.shape[0]
    idx = torch.argmax(vector).item()
    # idx = idx + 1 if idx < vector_size - 1 else idx
    normalized_coordinate = idx / (vector_size - 1)
    return int(normalized_coordinate * image_dimension_size)


def decode_output_tensor(y: torch.Tensor, filter_by_objectness_score: bool = False):
    n_vectors, vector_size = y.shape
    n_objects_vector = y[0, :]
    n_objects = torch.argmax(n_objects_vector)

    bbox_vector_size = vector_size - 80
    _coord_step = bbox_vector_size // 4

    objects = y[1:, :]
    h, w = y.shape[0], y.shape[1]

    bboxes = []
    categories = []
    objectness_scores = []
    for o in objects:
        objectness_score = o[-1]
        if filter_by_objectness_score and objectness_score < 0.5:
            continue
        bbox_raw = o[: (len(o) - 80)]
        x1 = _decode_coordinate_vector(bbox_raw[:_coord_step], w)
        y1 = _decode_coordinate_vector(bbox_raw[_coord_step : 2 * _coord_step], h)
        x2 = _decode_coordinate_vector(bbox_raw[2 * _coord_step : 3 * _coord_step], w)
        y2 = _decode_coordinate_vector(bbox_raw[3 * _coord_step : 4 * _coord_step], h)

        if all(x == 0 for x in [x1, y1, x2, y2]):
            continue
        bbox = [x1, y1, x2, y2]
        category = torch.argmax(o[(len(o) - 80) :])
        bboxes.append(bbox)
        categories.append(category)
        objectness_scores.append(objectness_score)

    # Sort by objectness score
    sorted_indices = sorted(
        range(len(objectness_scores)), key=lambda k: objectness_scores[k], reverse=True
    )
    bboxes = [bboxes[i] for i in sorted_indices]
    categories = [categories[i] for i in sorted_indices]
    objectness_scores = [objectness_scores[i] for i in sorted_indices]
    return bboxes[:n_objects], categories[:n_objects], objectness_scores[:n_objects]


def write_image_with_output(
    temp_out: torch.Tensor,
    validation_image: torch.Tensor,
    sub_dir: str = "any",
):
    bboxes, categories, objectness_scores = decode_output_tensor(temp_out.squeeze(0))

    validation_img = validation_image.squeeze(0).detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    image = (validation_img.numpy() * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for bbox, category in zip(bboxes, categories):
        x1, y1, x2, y2 = bbox
        print("Decoded:", x1, y1, x2, y2)
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