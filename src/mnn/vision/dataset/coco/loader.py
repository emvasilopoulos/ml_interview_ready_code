import abc
import random
from typing import Any, Dict, List, Tuple
import json
import os
import pathlib

import torch
import cv2
import numpy as np
import numpy.typing as npt

import mnn.vision.dataset.object_detection.preprocessing
import mnn.vision.image_size


class RawCOCOAnnotationsParser:

    def __init__(self, annotations_path: pathlib.Path, split: str):
        self.annotations_path = annotations_path
        self.split = split
        self.objects_by_image_id: Dict[str, List[Any]] = {}
        self.data_to_store = {}

    def parse_data(self):
        self._load_json_data()
        self.group_objects_by_image_id()
        self.data_to_store["annotations_grouped_by_image_id"] = self.objects_by_image_id
        self.data_to_store["images"] = self.images

    def write_data(self, output_path: pathlib.Path):
        with open(output_path, "w") as f:
            json.dump(self.data_to_store, f)

    def _load_json_data(self):
        with open(self.annotations_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]

    def group_objects_by_image_id(self):
        for annotation in self.annotations:
            image_id = annotation["image_id"]
            if image_id not in self.objects_by_image_id:
                self.objects_by_image_id[image_id] = []
            self.objects_by_image_id[image_id].append(annotation)
        return self.objects_by_image_id


class BaseCOCODatasetGrouped(torch.utils.data.Dataset):
    preprocessor = mnn.vision.dataset.object_detection.preprocessing.FadedBboxMasks

    @abc.abstractmethod
    def get_year(self) -> int:
        pass

    @abc.abstractmethod
    def get_type(self) -> str:
        """
        one of:
        - captions
        - instances
        - person_keypoints
        """
        pass

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
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.expected_image_size = expected_image_size
        year = self.get_year()
        coco_type = self.get_type()

        self.images_dir = data_dir / f"{split}{year}"
        self.annotations_path = (
            data_dir
            / "annotations"
            / f"{coco_type}_{split}{year}_grouped_by_image_id.json"
        )

        self.images = None
        self.annotations: Dict[str, Any] = None

        self._load_json_data()
        self.desired_classes = classes

    def _load_json_data(self):
        with open(self.annotations_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations_grouped_by_image_id"]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image_sample_i = self.images[idx]
        filename = image_sample_i["file_name"]
        sample_i_id = str(image_sample_i["id"])
        annotations = self.annotations.get(sample_i_id, [])

        # Prepare input
        img = cv2.imread(self.images_dir / filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        padding_percent = random.random()
        img_tensor = self.preprocessor.cv2_image_to_tensor(img)
        img_tensor = self.preprocessor.preprocess_image(
            img_tensor, self.expected_image_size, padding_percent=padding_percent
        )
        # Prepare output
        output0 = self.get_output(img, annotations, padding_percent=padding_percent)
        return img_tensor, output0

    @abc.abstractmethod
    def get_output(self, img: np.ndarray, annotations: Dict[str, Any]) -> torch.Tensor:
        pass


class BaseCOCODatasetInstances(BaseCOCODatasetGrouped):

    def get_output(
        self,
        img: npt.NDArray[np.uint8],
        annotations: Dict[str, Any],
        padding_percent: float = 0,
    ) -> torch.Tensor:
        bboxes = []
        categories = []
        img_h, img_w = img.shape[0], img.shape[1]
        for annotation in annotations:
            category = int(annotation["category_id"])
            if (
                self.desired_classes is not None
                and category not in self.desired_classes
            ):
                continue
            x1, y1, w, h = annotation["bbox"]
            bboxes.append([x1 / img_w, y1 / img_h, w / img_w, h / img_h])
            categories.append(annotation["category_id"])

        bboxes_as_mask = self.preprocessor.bboxes_to_mask(
            torch.Tensor(bboxes).float(), torch.Size((img_h, img_w))
        )
        bboxes_as_mask = self.preprocessor.adjust_tensor_dimensions(
            bboxes_as_mask, self.expected_image_size, padding_percent=padding_percent
        )

        # TODO - support categories somehow

        return bboxes_as_mask


class COCODatasetInstances2017(BaseCOCODatasetInstances):

    def get_year(self) -> int:
        return 2017

    def get_type(self) -> str:
        return "instances"
