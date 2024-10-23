import abc
import random
from typing import Any, Dict, List, Tuple
import json
import pathlib

import torch
import cv2
import numpy as np
import numpy.typing as npt
import torchvision

import mnn.vision.image_size
import mnn.vision.process_input.format
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.process_input.normalize
import mnn.vision.process_input.normalize.basic
from mnn.vision.process_input.pipeline import ProcessInputPipeline
from mnn.vision.process_input.reader import read_image_torchvision
import mnn.vision.process_output
import mnn.vision.process_output.object_detection
import mnn.vision.process_output.object_detection.rectangles_to_mask


class RawCOCOAnnotationsParser:

    def __init__(self, annotations_path: pathlib.Path):
        self.annotations_path = annotations_path
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

        self.input_pipeline = ProcessInputPipeline(
            dtype_converter=torchvision.transforms.ConvertImageDtype(torch.float32),
            normalize=mnn.vision.process_input.normalize.basic.NORMALIZE,
        )

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
        img_tensor = read_image_torchvision(self.images_dir / filename)
        img_tensor = self.input_pipeline(img_tensor)

        padding_percent = random.random()
        pad_value = random.random()
        img_tensor = mnn_resize_fixed_ratio.transform(
            img_tensor,
            self.expected_image_size,
            padding_percent=padding_percent,
            pad_value=pad_value,
        )

        # Prepare output
        output0 = self.get_output_tensor(
            img_tensor, annotations, padding_percent=padding_percent
        )
        print(output0.shape)
        return img_tensor, output0

    @abc.abstractmethod
    def get_output(self, img: np.ndarray, annotations: Dict[str, Any]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_output_tensor(
        self, img: torch.Tensor, annotations: Dict[str, Any]
    ) -> torch.Tensor:
        pass


class COCODatasetInstances2017(BaseCOCODatasetGrouped):

    def get_year(self) -> int:
        return 2017

    def get_type(self) -> str:
        return "instances"
