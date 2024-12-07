import abc
import pathlib

import torch

from mnn.vision.dataset.coco.torch_dataset_csv import BaseCOCODatasetGroupedCsv
import mnn.vision.image_size
import mnn.logging
import mnn.vision.dataset.coco.experiments.ordinal.indices as mnn_ordinal

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

    ORDINAL_EXPANSION = 10
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

        self.vector_indices = mnn_ordinal.VectorIndices(
            self.output_shape.width, self.N_CLASSES
        )

    @abc.abstractmethod
    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        pass

    #####################################################################################

    def _expand_left(self, vector: torch.Tensor, idx: int, expand_by: int = -1):
        if expand_by == -1:
            expand_by = self.ORDINAL_EXPANSION
        left_side = idx - expand_by
        if left_side < 0:
            left_side = 0
        d = idx - left_side
        for i in range(1, d):
            probability = i / expand_by
            vector[idx - (d - i)] = probability

        return vector

    def _expand_right(self, vector: torch.Tensor, idx: int, expand_by: int = -1):
        if expand_by == -1:
            expand_by = self.ORDINAL_EXPANSION
        vector_size = len(vector)
        # Right side
        right_side = idx + expand_by
        if right_side > vector_size:
            right_side = vector_size
        d = right_side - idx
        for i in range(1, d):
            probability = i / expand_by
            vector[idx + (d - i)] = probability
        return vector

    def _create_object_vector(self, bbox: list[float], category: int, vector_size: int):
        """expecting bbox as x1, y1, x2, y2"""
        vector = torch.zeros(vector_size)

        x1_coord_norm, y1_coord_norm, x2_coord_norm, y2_coord_norm = bbox
        # Bbox
        coordinate_span_of_indices_length = self.vector_indices.coord_len
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

        # Objectness
        idx = self.vector_indices.objectness_idx
        vector[idx] = 1

        # Category
        idx = category
        category_vector = torch.zeros(self.N_CLASSES)
        category_vector[idx] = 1
        # category_vector = self._expand_left(category_vector, idx)
        # category_vector = self._expand_right(category_vector, idx)
        vector[self.vector_indices.classes_start_idx :] = category_vector
        return vector

    def _decode_coordinate_vector(
        self, vector: torch.Tensor, image_dimension_size: int
    ) -> int:
        """
        Expecting tensors: (vector_size,)
        """
        if len(vector.shape) != 1:
            raise ValueError("The vector should be 1-dimensional")
        vector_size = vector.shape[0]
        idx = torch.argmax(vector).item()
        normalized_coordinate = idx / (vector_size - 1)
        return int(normalized_coordinate * image_dimension_size)

    def _decode_coordinate_vector_norm(self, vector: torch.Tensor) -> int:
        """
        Expecting tensors: (vector_size,)
        """
        if len(vector.shape) != 1:
            raise ValueError("The vector should be 1-dimensional")
        vector_size = vector.shape[0]
        idx = torch.argmax(vector).item()
        return idx / (vector_size - 1)

    def _decode_coordinate_vector_batch(
        self, vector: torch.Tensor, image_dimension_size: int
    ) -> torch.Tensor:
        """
        Expecting tensors: (batch_size, n_vectors, vector_size)
        """
        vector_size = vector.shape[2]
        indices = torch.argmax(vector, dim=2)
        normalized_coordinate = indices / (vector_size - 1)
        return normalized_coordinate * image_dimension_size

    def _decode_coordinate_vector_norm_batch(self, vector: torch.Tensor) -> int:
        vector_size = vector.shape[2]
        indices = torch.argmax(vector, dim=2)
        return indices / (vector_size - 1)
