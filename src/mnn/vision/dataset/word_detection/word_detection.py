from typing import List
import json
import pathlib

import torch.utils.data

from mnn.vision.dataset import utilities


class WordDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: pathlib.Path):
        self.dataset_path = dataset_path

        self.images_path = dataset_path / "screenshots"
        self.images_paths: List[pathlib.Path] = list(self.images_path.glob("*.jpg"))

        self.labels_path = dataset_path / "rendered_bboxes_json_files"
        self.labels_paths: List[pathlib.Path] = list(self.labels_path.glob("*.json"))

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path: pathlib.Path = self.images_paths[idx]
        label_path: pathlib.Path = self.labels_paths[idx]

        image = utilities.load_image_as_tensor(image_path)

        with label_path.open(mode="r") as f:
            label = json.load(f)

        return image, label
