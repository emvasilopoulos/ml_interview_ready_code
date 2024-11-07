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
]


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
                df_cropped = pd.concat([new_df, df_cropped], ignore_index=True)
        self.df_cropped_groups_by_image_id = df_cropped.groupby("image_id")
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

        x1 = int(data_for_image["start_x.crop"].values[0])
        y1 = int(data_for_image["start_y.crop"].values[0])
        x2 = int(data_for_image["end_x.crop"].values[0])
        y2 = int(data_for_image["end_y.crop"].values[0])
        img_tensor = img_tensor[:, y1:y2, x1:x2]

        x1s = data_for_image["x1_norm.crop"].values
        y1s = data_for_image["y1_norm.crop"].values
        ws = data_for_image["w_norm.crop"].values
        hs = data_for_image["h_norm.crop"].values

        # Filter bboxes that are outside the crop
        x1_list = []
        y1_list = []
        w_list = []
        h_list = []
        for i in range(len(x1s)):
            x1 = x1s[i]
            if x1 < 0 or x1 > 1:
                continue
            y1 = y1s[i]
            if y1 < 0 or y1 > 1:
                continue
            w = ws[i]
            if x1 + w > 1:
                w = 1 - x1
            h = hs[i]
            if y1 + h > 1:
                h = 1 - y1
            x1_list.append(x1)
            y1_list.append(y1)
            w_list.append(w)
            h_list.append(h)

        categories = data_for_image["category_id"].values
        annotations = self._prepare_annotations(
            x1_list, y1_list, w_list, h_list, categories
        )
        return img_tensor, annotations

    def get_pair(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if idx < len(self.df_original_groups_by_image_id):
            return self._get_pair_original(idx)
        else:
            return self._get_pair_cropped(
                idx - len(self.df_original_groups_by_image_id)
            )


class COCODatasetInstances2017(BaseCOCODatasetGroupedCsv):

    def get_year(self) -> int:
        return 2017

    def get_type(self) -> str:
        return "instances"
