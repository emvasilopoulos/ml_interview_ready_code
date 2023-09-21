import os
from typing import Tuple
import cv2
import numpy as np
import logging
from torch.utils.data import Dataset

from src.checkers import is_image_file, is_labels_file
from src.images.dataset.utilities import read_int_label_from_txt

logger = logging.getLogger("Images-Dataset-Logger")
logger.setLevel(logging.WARNING)


class ImagesClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        images_format_suffix: str = ".jpg",
        labels_format_suffix: str = ".txt",
        images_subdirectory: str = "images",
        labels_subdirectory: str = "labels",
    ) -> None:
        super().__init__()

        # Checks
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Path: '{dataset_path}' does not exist...")

        images_path = os.path.join(dataset_path, images_subdirectory)
        labels_path = os.path.join(dataset_path, labels_subdirectory)
        #
        self.images_and_labels = [
            (
                os.path.join(images_path, image),
                os.path.join(
                    labels_path,
                    image.replace(images_format_suffix, labels_format_suffix),
                ),
            )
            if is_image_file(image)
            and os.path.exists(
                image.replace(images_format_suffix, labels_format_suffix)
            )
            else logger.warn(
                f'WARNING\n1) Check if image for sample "{image}" is {images_format_suffix}\n2) Check if label "{image.replace(images_format_suffix, labels_format_suffix)}" exists...'
            )
            for image in os.listdir(images_path)
        ]

    def __len__(self):
        return len(self.images_and_labels)

    def __getitem__(self, index) -> Tuple[np.ndarray, int | str]:
        img = cv2.imread(self.images_and_labels[index][0])
        label = read_int_label_from_txt(self.images_and_labels[index][1])
        return img, label
