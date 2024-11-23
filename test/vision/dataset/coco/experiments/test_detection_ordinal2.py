import pathlib
import random
import unittest
import logging

import torch

import mnn.vision.dataset.coco.experiments.detection_ordinal2 as mnn_detection_ordinal
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.process_output.object_detection.bbox_mapper as mnn_bbox_mapper
import mnn.vision.image_size
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__, loglevel=logging.INFO)


class TestCOCOInstances2017Ordinal(unittest.TestCase):

    def setUp(self) -> None:

        dataset_dir = pathlib.Path("/home/manos/ml_interview_ready_code/data/")

        self.expected_image_sizeW = mnn.vision.image_size.ImageSize(576, 676)
        self.datasetW = mnn_detection_ordinal.COCOInstances2017Ordinal(
            data_dir=dataset_dir,
            split="val",
            expected_image_size=self.expected_image_sizeW,
        )
        self.imageW, self.targetW = self.datasetW[0]

        self.expected_image_sizeH = mnn.vision.image_size.ImageSize(676, 576)
        self.datasetH = mnn_detection_ordinal.COCOInstances2017Ordinal(
            data_dir=dataset_dir,
            split="val",
            expected_image_size=self.expected_image_sizeH,
        )
        self.imageH, self.targetH = self.datasetH[0]

        self.expected_image_sizeSq = mnn.vision.image_size.ImageSize(576, 576)
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

    def _test_create_object_vector(
        self, dataset: mnn_detection_ordinal.COCOInstances2017Ordinal
    ):
        x1 = 0
        y1 = 0
        x2 = 0.1
        y2 = 0.1

        category = 0

        vector = dataset._create_object_vector(
            bbox=[x1, y1, x2, y2],
            category=category,
            vector_size=dataset.expected_image_size.width,
        )
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

    def test_create_object_vector(self):
        LOGGER.info("test_create_object_vector")
        self._test_create_object_vector(self.datasetH)
        self._test_create_object_vector(self.datasetW)
        self._test_create_object_vector(self.datasetSq)

    def _test_decode_output_tensor(
        self, dataset: mnn_detection_ordinal.COCOInstances2017Ordinal
    ):
        output_mask = torch.zeros(
            (dataset.output_shape.height, dataset.output_shape.width)
        )

        """ TEST 1 """
        # Prepare
        coord_len = dataset.bbox_vector_size // 4
        # x1 = 0
        output_mask[0, 0] = 1
        output_mask[0, 1] = 0.75
        output_mask[0, 2] = 0.5
        output_mask[0, 3] = 0.25
        # y1 = 0
        output_mask[0, coord_len] = 1
        output_mask[0, coord_len + 1] = 0.75
        output_mask[0, coord_len + 2] = 0.5
        output_mask[0, coord_len + 3] = 0.25
        # w = 0.1
        output_mask[0, 2 * coord_len + int(0.1 * coord_len)] = 1
        output_mask[0, 2 * coord_len + int(0.1 * coord_len) + 1] = 0.75
        output_mask[0, 2 * coord_len + int(0.1 * coord_len) + 2] = 0.5
        output_mask[0, 2 * coord_len + int(0.1 * coord_len) + 3] = 0.25
        # h = 0.1
        output_mask[0, 3 * coord_len + int(0.1 * coord_len)] = 1
        output_mask[0, 3 * coord_len + int(0.1 * coord_len) + 1] = 0.75
        output_mask[0, 3 * coord_len + int(0.1 * coord_len) + 2] = 0.5
        output_mask[0, 3 * coord_len + int(0.1 * coord_len) + 3] = 0.25
        # category = 4
        output_mask[0, dataset.bbox_vector_size + 1] = 0.25
        output_mask[0, dataset.bbox_vector_size + 2] = 0.5
        output_mask[0, dataset.bbox_vector_size + 3] = 0.75
        output_mask[0, dataset.bbox_vector_size + 4] = 1
        output_mask[0, dataset.bbox_vector_size + 5] = 0.75
        output_mask[0, dataset.bbox_vector_size + 6] = 0.5
        output_mask[0, dataset.bbox_vector_size + 7] = 0.25

        bboxes, categories, scores = dataset.decode_output_tensor(output_mask)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(len(categories), 1)
        self.assertEqual(len(scores), 1)

        bbox = bboxes[0]
        self.assertAlmostEqual(bbox[0].item(), 0)
        self.assertAlmostEqual(bbox[1].item(), 0)
        self.assertAlmostEqual(
            bbox[2].item() / dataset.expected_image_size.width, 0.1, delta=0.008
        )
        self.assertAlmostEqual(
            bbox[3].item() / dataset.expected_image_size.height, 0.1, delta=0.008
        )
        self.assertEqual(categories[0].item(), 4)
        self.assertEqual(scores[0].item(), 1)

        """ TEST 2 """
        # x1 = 0 in grid 2 <=> Sx=2, Sy=0
        output_mask[2, 0] = 1
        output_mask[2, 1] = 0.75
        output_mask[2, 2] = 0.5
        output_mask[2, 3] = 0.25
        # y1 = 0 in grid 2 <=> Sx=2, Sy=0
        output_mask[2, coord_len] = 1
        output_mask[2, coord_len + 1] = 0.75
        output_mask[2, coord_len + 2] = 0.5
        output_mask[2, coord_len + 3] = 0.25
        # w = 0.1
        output_mask[2, 2 * coord_len + int(0.1 * coord_len)] = 1
        output_mask[2, 2 * coord_len + int(0.1 * coord_len) + 1] = 0.75
        output_mask[2, 2 * coord_len + int(0.1 * coord_len) + 2] = 0.5
        output_mask[2, 2 * coord_len + int(0.1 * coord_len) + 3] = 0.25
        # h = 0.1
        output_mask[2, 3 * coord_len + int(0.1 * coord_len)] = 1
        output_mask[2, 3 * coord_len + int(0.1 * coord_len) + 1] = 0.75
        output_mask[2, 3 * coord_len + int(0.1 * coord_len) + 2] = 0.5
        output_mask[2, 3 * coord_len + int(0.1 * coord_len) + 3] = 0.25
        # category = 4
        output_mask[2, dataset.bbox_vector_size + 2] = 0.25
        output_mask[2, dataset.bbox_vector_size + 3] = 0.5
        output_mask[2, dataset.bbox_vector_size + 4] = 0.75
        output_mask[2, dataset.bbox_vector_size + 5] = 1
        output_mask[2, dataset.bbox_vector_size + 6] = 0.75
        output_mask[2, dataset.bbox_vector_size + 7] = 0.5
        output_mask[2, dataset.bbox_vector_size + 8] = 0.25

        bboxes, categories, scores = dataset.decode_output_tensor(output_mask)
        self.assertEqual(len(bboxes), 2)
        self.assertEqual(len(categories), 2)
        self.assertEqual(len(scores), 2)

        bbox = bboxes[1]
        dataset.image_grid_S.width
        self.assertAlmostEqual(
            bbox[0].item() / dataset.expected_image_size.width,
            dataset.image_grid_S.width * 2 / dataset.expected_image_size.width,
        )
        self.assertAlmostEqual(
            bbox[1].item() / dataset.expected_image_size.height,
            0,
        )
        self.assertAlmostEqual(
            bbox[2].item() / dataset.expected_image_size.width, 0.1, delta=0.008
        )
        self.assertAlmostEqual(
            bbox[3].item() / dataset.expected_image_size.height, 0.1, delta=0.008
        )
        self.assertEqual(categories[1].item(), 5)
        self.assertEqual(scores[1].item(), 1)

    def test_decode_output_tensor(self):
        LOGGER.info("test_decode_output_tensor")
        self._test_decode_output_tensor(self.datasetH)
        self._test_decode_output_tensor(self.datasetW)
        self._test_decode_output_tensor(self.datasetSq)

    def _test_get_output_tensor(
        self, dataset: mnn_detection_ordinal.COCOInstances2017Ordinal
    ):
        current_image_size = mnn.vision.image_size.ImageSize(width=100, height=100)
        expected_image_size = mnn.vision.image_size.ImageSize(width=100, height=120)
        fixed_ratio_components = mnn_resize_fixed_ratio.calculate_new_tensor_dimensions(
            current_image_size, expected_image_size
        )
        padding_percent = random.random()
        bboxes = [(0, 0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)]
        categories = [4, 5]
        scores = [1, 1]
        annotations = [
            {
                "normalized_bbox": bboxes[0],
                "category_id": 4,
            },
            {
                "normalized_bbox": bboxes[1],
                "category_id": 5,
            },
        ]

        output_mask = dataset.get_output_tensor(
            annotations, fixed_ratio_components, padding_percent
        )
        self.assertEqual(output_mask.shape[0], dataset.output_shape.height)
        self.assertEqual(output_mask.shape[1], dataset.output_shape.width)
        bboxes_out, categories_out, scores_out = dataset.decode_output_tensor(
            output_mask
        )

        bbox_mapped0 = mnn_bbox_mapper.translate_norm_bbox_to_padded_image(
            bboxes[0], fixed_ratio_components, padding_percent, expected_image_size
        )
        bbox_mapped1 = mnn_bbox_mapper.translate_norm_bbox_to_padded_image(
            bboxes[1], fixed_ratio_components, padding_percent, expected_image_size
        )
        bboxes_mapped = [bbox_mapped0, bbox_mapped1]
        for bbox, bbox_out in zip(bboxes_mapped, bboxes_out):
            x = bbox[0] * expected_image_size.width
            y = bbox[1] * expected_image_size.height
            w = bbox[2] * expected_image_size.width
            h = bbox[3] * expected_image_size.height
            print(x, y, w, h)
            print(bbox_out)
            torch.testing.assert_close(torch.Tensor([x, y, w, h]), bbox_out)
        torch.testing.assert_close(torch.Tensor(categories), categories_out)
        torch.testing.assert_close(torch.Tensor(scores), scores_out)

    def test_get_output_tensor(self):
        LOGGER.info("test_get_output_tensor")
        self._test_get_output_tensor(self.datasetH)
        self._test_get_output_tensor(self.datasetW)
        self._test_get_output_tensor(self.datasetSq)


if __name__ == "__main__":
    unittest.main()
