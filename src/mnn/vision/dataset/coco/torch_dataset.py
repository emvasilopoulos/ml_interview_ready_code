import abc
import random
from typing import Any, Dict, List, Optional, Tuple
import json
import pathlib

import torch
import cv2
import numpy as np
import numpy.typing as npt
import torchvision

import mnn.vision.process_input.dimensions.pad as mnn_pad
import mnn.vision.process_input.dimensions.resize as mnn_resize
import mnn.vision.image_size
import mnn.vision.process_input.format
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio
import mnn.vision.process_input.normalize.basic
import mnn.vision.process_input.pipeline
import mnn.vision.process_input.reader
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

        self.input_pipeline = mnn.vision.process_input.pipeline.ProcessInputPipeline(
            dtype_converter=mnn.vision.process_input.pipeline.MyConvertImageDtype(
                torch.float32
            ),
            normalize=mnn.vision.process_input.normalize.basic.NORMALIZE,
        )

    def _load_json_data(self):
        with open(self.annotations_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations_grouped_by_image_id"]

    def __len__(self) -> int:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_tensor, annotations = self.get_pair(idx)

        # Calculate new dimensions & resize only
        current_image_size = mnn.vision.image_size.ImageSize(
            width=img_tensor.shape[2], height=img_tensor.shape[1]
        )
        fixed_ratio_components = mnn_resize_fixed_ratio.calculate_new_tensor_dimensions(
            current_image_size, self.expected_image_size
        )
        img_tensor = mnn_resize.resize_image(
            img_tensor,
            fixed_ratio_components.resize_height,
            fixed_ratio_components.resize_width,
        )

        # Random padding that both input & output must know about
        padding_percent = random.random()
        pad_value = random.random()

        # Prepare output based on expected image size & padding that will be applied in image
        output0 = self.get_output_tensor(
            annotations,
            fixed_ratio_components,
            padding_percent=padding_percent,
            current_image_size=current_image_size,
        )

        # Apply padding to image
        img_tensor = mnn_pad.pad_image(
            img_tensor,
            fixed_ratio_components.pad_dimension,
            fixed_ratio_components.expected_dimension_size,
            padding_percent,
            pad_value,
        )
        return img_tensor, output0

    @abc.abstractmethod
    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
        current_image_size: Optional[mnn.vision.image_size.ImageSize] = None,
    ) -> torch.Tensor:
        pass


class COCODatasetInstances2017(BaseCOCODatasetGrouped):

    def get_year(self) -> int:
        return 2017

    def get_type(self) -> str:
        return "instances"
