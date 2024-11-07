from typing import Any, Dict, List, Tuple
import json
import pathlib

import torch

from mnn.vision.dataset.coco.base import BaseCOCODatasetGrouped
import mnn.vision.image_size
import mnn.vision.process_input.format
import mnn.vision.process_input.normalize.basic
import mnn.vision.process_input.pipeline
import mnn.vision.process_input.reader
import mnn.vision.process_output.object_detection
import mnn.vision.process_output.object_detection.rectangles_to_mask
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)


class BaseCOCODatasetGroupedJson(BaseCOCODatasetGrouped):

    def __init__(
        self,
        data_dir: pathlib.Path,
        split: str,
        expected_image_size: mnn.vision.image_size.ImageSize,
        classes: List[str] = None,
    ):
        """
        Args:
            data_dir (str): path to the COCO dataset directory.
            split (str): the split to use, either 'train' or 'val'.
            transforms (Compose): a composition of torchvision.transforms to apply to the images.
        """
        super().__init__(data_dir, split, expected_image_size, classes)

    def _read_annotations(self, annotations_dir, coco_type, split, year):
        self.annotations_path = (
            annotations_dir / f"{coco_type}_{split}{year}_grouped_by_image_id.json"
        )
        with open(self.annotations_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations_grouped_by_image_id"]

    def _define_length(self) -> int:
        return len(self.images)

    def get_pair(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Get image and annotations
        image_sample_i = self.images[idx]
        filename = image_sample_i["file_name"]
        sample_i_id = str(image_sample_i["id"])
        annotations = self.annotations.get(sample_i_id, [])

        self._current_sample = filename
        # Read image as tensor
        img_tensor = mnn.vision.process_input.reader.read_image_torchvision(
            self.images_dir / filename
        )
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        img_tensor = self.input_pipeline(img_tensor)
        img_w = img_tensor.shape[2]
        img_h = img_tensor.shape[1]

        # Normalize bounding boxes
        for annotation in annotations:
            x1, y1, w, h = annotation["bbox"]
            normalized_bbox = [x1 / img_w, y1 / img_h, w / img_w, h / img_h]
            annotation["normalized_bbox"] = normalized_bbox
        return img_tensor, annotations
