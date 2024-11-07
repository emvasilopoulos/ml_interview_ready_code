import abc
import random
from typing import Any, Dict, List, Optional, Tuple
import pathlib


import torch

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
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)


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

    @abc.abstractmethod
    def get_pair(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def _read_annotations(
        self, annotations_dir: pathlib.Path, coco_type: str, split: str, year: str
    ):
        pass

    @abc.abstractmethod
    def _define_length(self) -> int:
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
        annotations_dir = data_dir / "annotations"

        self.desired_classes = classes

        self.input_pipeline = mnn.vision.process_input.pipeline.ProcessInputPipeline(
            dtype_converter=mnn.vision.process_input.pipeline.MyConvertImageDtype(
                torch.float32
            ),
            normalize=mnn.vision.process_input.normalize.basic.NORMALIZE,
        )

        self._read_annotations(annotations_dir, coco_type, split, year)

    def _read_image(self, image_id: int) -> torch.Tensor:
        filename = self._image_file_name_from_id(image_id)
        img_tensor = mnn.vision.process_input.reader.read_image_torchvision(
            self.images_dir / filename
        )
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        return self.input_pipeline(img_tensor)

    def _image_file_name_from_id(self, image_id: int) -> str:
        return f"{image_id:012}.jpg"

    def __len__(self) -> int:
        return self._define_length()

    def _random_percentage(self):
        side = random.randint(0, 1)
        if side == 0:
            return random.random() * 0.14
        else:
            return random.random() * 0.14 + 0.86

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
        padding_percent = self._random_percentage()
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
        raise NotImplementedError()


class COCODatasetInstances2017(BaseCOCODatasetGrouped):

    def get_year(self) -> int:
        return 2017

    def get_type(self) -> str:
        return "instances"
