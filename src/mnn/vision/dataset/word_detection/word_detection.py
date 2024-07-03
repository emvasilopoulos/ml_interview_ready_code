from typing import List
import abc
import json
import pathlib

import numpy as np
import torch.utils.data

import mnn.vision.dataset.utilities as vision_utilities
import mnn.vision.dataset.word_detection.bounding_client_rect as bounding_client_rect
import mnn.vision.dataset.word_detection.label as label
import mnn.vision.dataset.word_detection.raw as raw

all_classes = [
    "other-symbol",
    "letter",
    "word",
    "digit",
    "number",
    "punctuation",
    "paragraph",
    "H1",
    "H2",
    "DIV",
    # ADD TABLES & MORE
    "figure",
    "image",
    "table",
]


class BaseWordDetectionDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(self, dataset_path: pathlib.Path):
        self.dataset_path = dataset_path

        self.images_path = dataset_path / "screenshots"
        self.images_paths: List[pathlib.Path] = list(self.images_path.glob("*.jpg"))

        self.labels_path = dataset_path / "rendered_bboxes_json_files"

        self.allowed_classes = self._get_dataset_classes()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, List[label.WordDetectionLabel]]:
        """
        Get the image and the labels for the given index.
        The user is responsible for converting the labels to the desired tensor format.
        """
        image_path: pathlib.Path = self.images_paths[idx]
        label_path: pathlib.Path = self.labels_path / f"{image_path.stem}.json"

        image = vision_utilities.load_image_as_tensor(
            image_path, normalize=True, as_bgr=False
        )
        labels = self._get_labels(label_path)

        return image, labels

    def get_debug_image(
        self, idx: int
    ) -> tuple[np.ndarray, List[label.WordDetectionLabel]]:
        image_path: pathlib.Path = self.images_paths[idx]
        label_path: pathlib.Path = self.labels_path / f"{image_path.stem}.json"

        image = vision_utilities.read_image(image_path)
        labels = self._get_labels(label_path)

        return image, labels

    def _get_labels(self, label_path: pathlib.Path) -> dict:
        with label_path.open(mode="r") as f:
            labels = json.load(f)
        return self._filter_labels(raw.RawDataset.from_dict(labels))

    @abc.abstractmethod
    def _filter_labels(self, labels: raw.RawDataset) -> List[label.WordDetectionLabel]:
        """
        Summary: Filter the detection objects according to the inheriting class requirements.
        Also transform the raw data to the label.WordDetectionLabel.

        Args:
            labels (raw.RawDataset): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            List[label.WordDetectionLabel]: _description_
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_dataset_classes(self) -> List[label.WordDetectionLabel]:
        raise NotImplementedError()

    def get_dataset_name(self) -> str:
        return self.__class__.__name__


class WordDetectionTextGroupsDataset(BaseWordDetectionDataset):

    def _filter_labels(self, labels: raw.RawDataset) -> List[label.WordDetectionLabel]:
        elements = []
        for element in labels.rendered_elements:
            if element.className is not None:
                if element.className not in self.allowed_classes:
                    continue
                class_id = self.allowed_classes[element.className]
            elif element.nodeName in self.allowed_classes:
                class_id = self.allowed_classes[element.nodeName]
            else:
                continue
            elements.append(
                label.get_label_from_bounding_client_rect(
                    class_id,
                    element.boundingClientRect,
                    normalized=True,
                    image_width=labels.body_shape.width,
                    image_height=labels.body_shape.height,
                )
            )
        return elements

    def _get_dataset_classes(self) -> dict:
        return {
            "H1": 0,
            "H2": 1,
            "paragraph": 2,
        }


if __name__ == "__main__":
    import cv2
    import random

    mypath = pathlib.Path("/Users/emlvasilopoulos/Datasets/word_det_train")
    dataset = WordDetectionTextGroupsDataset(mypath)
    idx = random.randint(0, len(dataset) - 1)
    x, y = dataset.get_debug_image(idx)
    print(x.shape)

    for bbox in y:
        cv2.rectangle(
            x,
            (int(bbox.x * x.shape[1]), int(bbox.y * x.shape[0])),
            (
                int((bbox.x + bbox.width) * x.shape[1]),
                int((bbox.y + bbox.height) * x.shape[0]),
            ),
            (0, 255, 0),
            2,
        )
    cv2.imshow("image", x)
    cv2.waitKey(0)
