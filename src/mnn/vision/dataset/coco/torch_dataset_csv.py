import abc
import random
from typing import Any, Dict, List, Optional, Tuple
import pathlib


import pandas as pd
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


HEADERS = [
    "id",
    "image_id",
    "category_id",
    "iscrowd",
    "x1",
    "y1",
    "w",
    "h",
    "x1_norm",
    "y1_norm",
    "w_norm",
    "h_norm",
    "image_width",
    "image_height",
    "start_x.crop",
    "start_y.crop",
    "end_x.crop",
    "end_y.crop",
    "x1_norm.crop",
    "y1_norm.crop",
    "w_norm.crop",
    "h_norm.crop",
    "rand_scheme",
]


def map_out_of_bounds_bbox_in_crop(x1: float, y1: float, w: float, h: float):
    """
    Returns -1, -1, -1, -1 if the bbox is completely out of the crop.
    """
    if x1 > 1:
        return (-1, -1, -1, -1), False
    if x1 + w < 0:
        return (-1, -1, -1, -1), False
    else:
        x1 = max(0, x1)

    if y1 > 1:
        return (-1, -1, -1, -1), False
    if y1 + h < 0:
        return (-1, -1, -1, -1), False
    else:
        y1 = max(0, y1)

    if x1 + w > 1:
        w = 1 - x1
    if y1 + h > 1:
        h = 1 - y1

    return (x1, y1, w, h), True


class BaseCOCODatasetGroupedCsv(BaseCOCODatasetGrouped):

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

    def _read_annotations(
        self, annotations_dir: pathlib.Path, coco_type: str, split: str, year: str
    ):
        # Read original annotations
        self.annotations_path = annotations_dir / f"{coco_type}_{split}{year}.csv"
        LOGGER.info(f"Original annotations: {self.annotations_path}")
        df_original = pd.read_csv(self.annotations_path)
        self.df_original_groups_by_image_id = df_original.groupby("image_id")
        self.original_groups_indexed = list(
            self.df_original_groups_by_image_id.groups.keys()
        )
        df_cropped = pd.DataFrame(columns=HEADERS)

        # Read annotations from crops
        self.annotations_schemes_paths = list(
            annotations_dir.glob(f"{coco_type}_{split}{year}_rand_scheme_*.csv")
        )
        LOGGER.info(f"Annotations from crops: {self.annotations_schemes_paths}")
        if len(self.annotations_schemes_paths):
            for scheme_path in self.annotations_schemes_paths:
                new_df = pd.read_csv(scheme_path)
                if "rand_scheme" not in new_df.columns:
                    new_df["rand_scheme"] = scheme_path.stem.split("_")[-1]
                    LOGGER.warning("Added 'rand_scheme' column to the dataframe")
                df_cropped = pd.concat([new_df, df_cropped], ignore_index=True)
        self.df_cropped_groups_by_image_id = df_cropped.groupby(
            ["image_id", "rand_scheme"]
        )
        self.cropped_groups_indexed = list(
            self.df_cropped_groups_by_image_id.groups.keys()
        )

    def _define_length(self) -> int:
        original_len = len(self.df_original_groups_by_image_id)
        cropped_len = len(self.df_cropped_groups_by_image_id)
        return original_len + cropped_len

    def _prepare_annotations(self, x1_norms, y1_norms, w_norms, h_norms, categories):
        return [
            {"normalized_bbox": [x1, y1, w, h], "category_id": category}
            for x1, y1, w, h, category in zip(
                x1_norms, y1_norms, w_norms, h_norms, categories
            )
        ]

    def _get_pair_original(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        group = self.original_groups_indexed[idx]
        data_for_image = self.df_original_groups_by_image_id.get_group(group)
        image_id = data_for_image["image_id"].values[0]
        img_tensor = self._read_image(image_id)

        x1s = data_for_image["x1_norm"].values
        y1s = data_for_image["y1_norm"].values
        ws = data_for_image["w_norm"].values
        hs = data_for_image["h_norm"].values
        categories = data_for_image["category_id"].values
        annotations = self._prepare_annotations(x1s, y1s, ws, hs, categories)
        return img_tensor, annotations

    def _get_pair_cropped(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        group = self.cropped_groups_indexed[idx]
        data_for_image = self.df_cropped_groups_by_image_id.get_group(group)
        image_id = data_for_image["image_id"].values[0]
        img_tensor = self._read_image(image_id)

        # Crop the image
        x1 = int(data_for_image["start_x.crop"].values[0])
        y1 = int(data_for_image["start_y.crop"].values[0])
        x2 = int(data_for_image["end_x.crop"].values[0])
        y2 = int(data_for_image["end_y.crop"].values[0])
        img_tensor = img_tensor[:, y1:y2, x1:x2]

        x1s = data_for_image["x1_norm.crop"].values
        y1s = data_for_image["y1_norm.crop"].values
        ws = data_for_image["w_norm.crop"].values
        hs = data_for_image["h_norm.crop"].values
        categories = data_for_image["category_id"].values

        # Filter bboxes that are outside the crop
        x1_list = []
        y1_list = []
        w_list = []
        h_list = []
        categories_list = []
        for i in range(len(x1s)):
            x1_crop = x1s[i]
            y1_crop = y1s[i]
            w_crop = ws[i]
            h_crop = hs[i]

            (x1_crop, y1_crop, w_crop, h_crop), is_inside = (
                map_out_of_bounds_bbox_in_crop(x1_crop, y1_crop, w_crop, h_crop)
            )
            if not is_inside:
                continue
            x1_list.append(x1_crop)
            y1_list.append(y1_crop)
            w_list.append(w_crop)
            h_list.append(h_crop)
            categories_list.append(categories[i])
        annotations = self._prepare_annotations(
            x1_list, y1_list, w_list, h_list, categories
        )
        return img_tensor, annotations

    def get_pair(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if idx < len(self.df_original_groups_by_image_id):
            img_tensor, annotations = self._get_pair_original(idx)
        else:
            img_tensor, annotations = self._get_pair_cropped(
                idx - len(self.df_original_groups_by_image_id)
            )
        if len(annotations) == 0:
            dtype = img_tensor.dtype
            shape = img_tensor.shape
            img_tensor = get_random_image(shape, dtype)
        return img_tensor, annotations


def get_random_image(shape, dtype):
    choice = random.randint(0, 2)
    if choice == 0:
        x = torch.ones(shape, dtype=dtype) * random.uniform(0, 1)
    elif choice == 1:
        x = torch.ones(shape, dtype=dtype)
        for channel in range(shape[0]):
            x[channel] *= random.uniform(0, 1)
    else:
        x = torch.rand(shape, dtype=dtype)
    return x


class COCODatasetInstances2017(BaseCOCODatasetGroupedCsv):

    def get_year(self) -> int:
        return 2017

    def get_type(self) -> str:
        return "instances"
