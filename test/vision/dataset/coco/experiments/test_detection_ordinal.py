import pathlib
import unittest

import torch

import mnn.vision.dataset.coco.experiments.detection_ordinal as mnn_detection_ordinal
import mnn.vision.image_size
import logging
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__, loglevel=logging.INFO)

class TestCOCOInstances2017Ordinal(unittest.TestCase):

    def setUp(self) -> None:

        dataset_dir = pathlib.Path(
            "/home/manos/ml_interview_ready_code/data/"
        )

        self.expected_image_sizeW = mnn.vision.image_size.ImageSize(480, 640)
        self.datasetW = mnn_detection_ordinal.COCOInstances2017Ordinal(
            data_dir=dataset_dir,
            split="val",
            expected_image_size=self.expected_image_sizeW,
        )
        self.imageW, self.targetW = self.datasetW[0]

        self.expected_image_sizeH = mnn.vision.image_size.ImageSize(640, 480)
        self.datasetH = mnn_detection_ordinal.COCOInstances2017Ordinal(
            data_dir=dataset_dir,
            split="val",
            expected_image_size=self.expected_image_sizeH,
        )
        self.imageH, self.targetH = self.datasetH[0]

        self.expected_image_sizeSq = mnn.vision.image_size.ImageSize(512, 512)
        self.datasetSq = mnn_detection_ordinal.COCOInstances2017Ordinal(
            data_dir=dataset_dir,
            split="val",
            expected_image_size=self.expected_image_sizeSq,
        )
        self.imageSq, self.targetSq = self.datasetSq[0]

    def _test_expand_left_right(
        self, dataset: mnn_detection_ordinal.COCOInstances2017Ordinal
    ):

        coordinate_span_of_indices_length = dataset.bbox_vector_size // 4
        # Use Case 1
        vector = torch.zeros(coordinate_span_of_indices_length)
        idx = 0
        vector[idx] = 1
        dataset._expand_left(vector, 0)
        dataset._expand_right(vector, 0)
        assert vector[idx - 3] == 0
        assert vector[idx - 2] == 0
        assert vector[idx - 1] == 0
        assert vector[idx] == 1
        assert vector[idx + 1] == 0.75
        assert vector[idx + 2] == 0.5
        assert vector[idx + 3] == 0.25

        # Use Case 2
        vector = torch.zeros(coordinate_span_of_indices_length)
        idx = coordinate_span_of_indices_length - 1
        vector[idx] = 1
        dataset._expand_right(vector, idx)
        dataset._expand_left(vector, idx)
        assert vector[idx - 3] == 0.25
        assert vector[idx - 2] == 0.5
        assert vector[idx - 1] == 0.75
        assert vector[idx] == 1
        with self.assertRaises(IndexError):
            assert vector[idx + 1] == 0
            assert vector[idx + 2] == 0
            assert vector[idx + 3] == 0

        # Use Case 3
        vector = torch.zeros(coordinate_span_of_indices_length)
        idx = coordinate_span_of_indices_length // 2
        vector[idx] = 1
        dataset._expand_left(vector, idx)
        dataset._expand_right(vector, idx)
        assert vector[idx - 3] == 0.25
        assert vector[idx - 2] == 0.5
        assert vector[idx - 1] == 0.75
        assert vector[idx] == 1
        assert vector[idx + 1] == 0.75
        assert vector[idx + 2] == 0.5
        assert vector[idx + 3] == 0.25

    def test_expand_left_right(self):
        LOGGER.info("test_expand_left_right")
        self._test_expand_left_right(self.datasetW)
        self._test_expand_left_right(self.datasetH)
        self._test_expand_left_right(self.datasetSq)

    def _test_create_object_vector(self, dataset: mnn_detection_ordinal.COCOInstances2017Ordinal):
        x1 = 0
        y1 = 0
        x2 = 0.1
        y2 = 0.1

        category = 0

        vector, is_big_enough = dataset._create_object_vector(bbox=[x1, y1, x2, y2], category=category, vector_size=dataset.expected_image_size.width)
        s = dataset.bbox_vector_size // 4
        # x1
        self.assertEqual(vector[0], 1)
        self.assertEqual(vector[1], 0.75)
        self.assertEqual(vector[2], 0.5)
        self.assertEqual(vector[3], 0.25)
        self.assertEqual(vector[4], 0)

        # y1
        self.assertEqual(vector[s - 4], 0)
        self.assertEqual(vector[s - 3], 0)
        self.assertEqual(vector[s - 2], 0)
        self.assertEqual(vector[s - 1], 0)
        self.assertEqual(vector[s], 1)
        self.assertEqual(vector[s + 1], 0.75)
        self.assertEqual(vector[s + 2], 0.5)
        self.assertEqual(vector[s + 3], 0.25)
        self.assertEqual(vector[s + 4], 0)

        # x2
        extra = round(x2 * s)
        self.assertEqual(vector[2 * s - 4 + extra], 0)
        self.assertEqual(vector[2 * s - 3 + extra], 0.25)
        self.assertEqual(vector[2 * s - 2 + extra], 0.5)
        self.assertEqual(vector[2 * s - 1 + extra], 0.75)
        self.assertEqual(vector[2 * s + extra], 1)
        self.assertEqual(vector[2 * s + 1 + extra], 0.75)
        self.assertEqual(vector[2 * s + 2 + extra], 0.5)
        self.assertEqual(vector[2 * s + 3 + extra], 0.25)
        self.assertEqual(vector[2 * s + 4 + extra], 0)

        # y2
        extra = round(y2 * s)
        self.assertEqual(vector[3 * s - 4 + extra], 0)
        self.assertEqual(vector[3 * s - 3 + extra], 0.25)
        self.assertEqual(vector[3 * s - 2 + extra], 0.5)
        self.assertEqual(vector[3 * s - 1 + extra], 0.75)
        self.assertEqual(vector[3 * s + extra], 1)
        self.assertEqual(vector[3 * s + 1 + extra], 0.75)
        self.assertEqual(vector[3 * s + 2 + extra], 0.5)
        self.assertEqual(vector[3 * s + 3 + extra], 0.25)
        self.assertEqual(vector[3 * s + 4 + extra], 0)

        # category
        self.assertEqual(vector[dataset.bbox_vector_size + category], 1)
        self.assertEqual(vector[dataset.bbox_vector_size + category + 1], 0.75)
        self.assertEqual(vector[dataset.bbox_vector_size + category + 2], 0.5)
        self.assertEqual(vector[dataset.bbox_vector_size + category + 3], 0.25)
        self.assertEqual(vector[dataset.bbox_vector_size + category + 4], 0)
        self.assertEqual(vector[dataset.bbox_vector_size + len(dataset.classes) - 1], 0)
        self.assertEqual(vector[dataset.bbox_vector_size + len(dataset.classes) - 2], 0)
        self.assertEqual(vector[dataset.bbox_vector_size + len(dataset.classes) - 3], 0)
        self.assertEqual(vector[dataset.bbox_vector_size + len(dataset.classes) - 4], 0)

        # Objectness
        self.assertEqual(vector[dataset.bbox_vector_size + len(dataset.classes)], 0.25)


    def test_create_object_vector(self):
        LOGGER.info("test_create_object_vector")
        self._test_create_object_vector(self.datasetH)
        self._test_create_object_vector(self.datasetW)
        self._test_create_object_vector(self.datasetSq)

    def _test_create_number_of_objects_vector(self, dataset: mnn_detection_ordinal.COCOInstances2017Ordinal):
        n_objects = 0
        vector = dataset._create_number_of_objects_vector(
            n_objects, dataset.vector_size
        )
        self.assertEqual(vector[0], 1)
        self.assertEqual(vector[1], 0.75)
        self.assertEqual(vector[2], 0.5)
        self.assertEqual(vector[3], 0.25)

        n_objects = dataset.vector_size - 1
        vector = dataset._create_number_of_objects_vector(
            n_objects, dataset.vector_size
        )
        self.assertEqual(vector[dataset.vector_size-4], 0.25)
        self.assertEqual(vector[dataset.vector_size-3], 0.5)
        self.assertEqual(vector[dataset.vector_size-2], 0.75)
        self.assertEqual(vector[dataset.vector_size-1], 1)

        n_objects = 10
        vector = dataset._create_number_of_objects_vector(
            n_objects, dataset.vector_size
        )
        self.assertEqual(vector[6], 0)
        self.assertEqual(vector[7], 0.25)
        self.assertEqual(vector[8], 0.5)
        self.assertEqual(vector[9], 0.75)
        self.assertEqual(vector[10], 1)
        self.assertEqual(vector[11], 0.75)
        self.assertEqual(vector[12], 0.5)
        self.assertEqual(vector[13], 0.25)
        self.assertEqual(vector[14], 0)

    def test_create_number_of_objects_vector(self):
        LOGGER.info("test_create_number_of_objects_vector")
        self._test_create_number_of_objects_vector(self.datasetH)
        self._test_create_number_of_objects_vector(self.datasetW)
        self._test_create_number_of_objects_vector(self.datasetSq)



if __name__ == "__main__":
    unittest.main()
