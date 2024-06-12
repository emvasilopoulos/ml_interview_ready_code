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


class ImageRegressionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        images_format_suffix: str = ".jpg",
        values_format_suffix: str = ".txt",
        images_subdirectory: str = "images",
        values_subdirectory: str = "values",
    ) -> None:
        super().__init__()

        # Checks
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Path: '{dataset_path}' does not exist...")

        images_path = os.path.join(dataset_path, images_subdirectory)
        values_path = os.path.join(dataset_path, values_subdirectory)
        #
        self.images_and_values = [
            (
                os.path.join(images_path, image),
                os.path.join(
                    values_path,
                    image.replace(images_format_suffix, values_format_suffix),
                ),
            )
            if is_image_file(image)
            and os.path.exists(
                image.replace(images_format_suffix, values_format_suffix)
            )
            else logger.warn(
                f'WARNING\n1) Check if image for sample "{image}" is {images_format_suffix}\n2) Check if label "{image.replace(images_format_suffix, values_format_suffix)}" exists...'
            )
            for image in os.listdir(images_path)
        ]

    def __len__(self):
        return len(self.images_and_values)

    def __getitem__(self, index) -> Tuple[np.ndarray, int | str]:
        img = cv2.imread(self.images_and_values[index][0])
        value = read_int_label_from_txt(self.images_and_values[index][1])
        return img, value
