import argparse
import os
import pathlib
from typing import List, Optional, Tuple

import cv2
import torch
import torch.utils.tensorboard
import tqdm

import mnn.logging
from mnn.losses import FocalLoss
from mnn.lr_scheduler import MyLRScheduler
import mnn.torch_utils as mnn_utils
from mnn.training_tools.parameters import get_params_grouped
from mnn.vision.config import load_hyperparameters_config
import mnn.vision.config as mnn_config
import mnn.vision.dataset.coco.experiments.detection_ordinal as mnn_ordinal
import mnn.vision.image_size
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.model as mnn_vit_model
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.config as mnn_vit_config

LOGGER = mnn.logging.get_logger(__name__)


def load_datasets(
    dataset_dir: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
    classes: Optional[List[int]] = None,
) -> Tuple[
    mnn_ordinal.COCOInstances2017Ordinal2,
    mnn_ordinal.COCOInstances2017Ordinal2,
]:
    val_dataset = mnn_ordinal.COCOInstances2017Ordinal2(
        dataset_dir,
        "val",
        expected_image_size,
    )
    return val_dataset


def write_image_with_output_of_experiment3(
    bboxes: List[Tuple[int, int, int, int]],
    categories: List[int],
    confidence_scores: List[float],
    validation_image: torch.Tensor,
    sub_dir: str = "any",
):
    validation_img = validation_image.detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    image = (validation_img.numpy() * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for bbox, category, confidence in zip(bboxes, categories, confidence_scores):
        if confidence <= 0.001:
            continue

        xc, yc, w, h = bbox
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = x1 + int(w)
        y2 = y1 + int(h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        category_no = category.item()
        cat = (
            f"{category_no} - {confidence:.3f}"
            if category_no < 1.0
            else f"{category_no}"
        )
        cv2.putText(
            image,
            cat,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # reverse mask
    os.makedirs(f"assessment_images/{sub_dir}", exist_ok=True)
    cv2.imwrite(f"assessment_images/{sub_dir}/bboxed_image.jpg", image)


if __name__ == "__main__":
    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
    )
    classes = None  # ALL CLASSES
    val_dataset = load_datasets(
        dataset_dir, mnn.vision.image_size.ImageSize(640, 480), classes
    )
    # Prepare validation image
    for i in range(3):
        validation_image, target = val_dataset[i]
        validation_image_unsqueezed = validation_image.unsqueeze(0)
        target_bboxes, target_categories, target_confidences = (
            val_dataset.decode_output_tensor(target)
        )
        write_image_with_output_of_experiment3(
            target_bboxes,
            target_categories,
            target_confidences,
            validation_image_unsqueezed.squeeze(0),
            f"validation_image_gt_{i}",
        )