import argparse
import pathlib
from typing import List, Optional, Tuple

import mnn.vision.image_size
import mnn.vision.dataset.coco.experiments.detection_ordinal as mnn_ordinal
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)


def load_datasets(
    dataset_dir: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
    classes: Optional[List[int]] = None,
) -> Tuple[
    mnn_ordinal.COCOInstances2017Ordinal,
    mnn_ordinal.COCOInstances2017Ordinal,
]:
    train_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir,
        "train",
        expected_image_size,
    )
    val_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir, "val", expected_image_size
    )
    return train_dataset, val_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=False,
        default="/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco",
    )
    args = parser.parse_args()

    LOGGER.info("------ LOADING ------")

    # DATASET
    LOGGER.info("dataset...")
    dataset_dir = pathlib.Path(args.dataset_dir)
    expected_image_size = mnn.vision.image_size.ImageSize(640, 480, 3)
    classes = None  # ALL CLASSES
    train_dataset, val_dataset = load_datasets(dataset_dir, expected_image_size)

    for i in range(len(train_dataset)):
        image, target = train_dataset[i]

    for i in range(len(val_dataset)):
        image, target = val_dataset[i]

    # Validation Image

