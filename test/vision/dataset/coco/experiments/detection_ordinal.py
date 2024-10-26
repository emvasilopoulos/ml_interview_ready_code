import pathlib
import unittest

import torch

import mnn.vision.dataset.coco.experiments.detection_ordinal as mnn_detection_ordinal
import mnn.vision.image_size


class TestCOCOInstances2017Ordinal(unittest.TestCase):

    def setUp(self) -> None:

        dataset_dir = pathlib.Path(
            "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
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
        self._test_expand_left_right(self.datasetW)
        self._test_expand_left_right(self.datasetH)
        self._test_expand_left_right(self.datasetSq)


if __name__ == "__main__":
    unittest.main()
