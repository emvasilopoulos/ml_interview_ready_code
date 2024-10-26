import argparse
import pathlib
from typing import List, Optional, Tuple
import logging

import torch
import torch.utils.tensorboard

from mnn.vision.config import load_hyperparameters_config
import mnn.vision.image_size
from mnn.vision.dataset.coco.training.train import (
    train_val,
)
import mnn.vision.dataset.coco.experiments.detection_fading_bboxes_in_mask as mnn_fading_bboxes_in_mask
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.model as mnn_vit_model
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.config as mnn_vit_config

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def load_datasets(
    dataset_dir: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
    classes: Optional[List[int]] = None,
) -> Tuple[
    mnn_fading_bboxes_in_mask.COCOInstances2017FBM,
    mnn_fading_bboxes_in_mask.COCOInstances2017FBM,
]:
    classes = None
    train_dataset = mnn_fading_bboxes_in_mask.COCOInstances2017FBM(
        dataset_dir, "train", expected_image_size, classes=classes
    )
    val_dataset = mnn_fading_bboxes_in_mask.COCOInstances2017FBM(
        dataset_dir, "val", expected_image_size, classes=classes
    )
    return train_dataset, val_dataset


def load_model(
    config_path: pathlib.Path, existing_model_path: Optional[pathlib.Path] = None
) -> mnn_vit_model.VitObjectDetectionNetwork:
    model_config, _, head_config = mnn_vit_config.load_model_config(config_path)
    model = mnn_vit_model.VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )
    if existing_model_path:
        model.load_state_dict(torch.load(existing_model_path))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=False,
        default="/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco",
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        required=False,
        default="/home/emvasilopoulos/projects/ml_interview_ready_code/experiments/coco/instances/experiment1/model.yaml",
    )
    parser.add_argument(
        "--hyperparameters-config-path",
        type=str,
        required=False,
        default="/home/emvasilopoulos/projects/ml_interview_ready_code/experiments/coco/instances/experiment1/hyperparameters.yaml",
    )
    parser.add_argument("--existing-model-path", type=str, required=False, default=None)
    args = parser.parse_args()

    LOGGER.info("------ LOADING ------")
    # MODEL
    model_config_path = pathlib.Path(args.model_config_path)
    if args.existing_model_path is not None:
        existing_model_path = pathlib.Path(args.existing_model_path)
        LOGGER.info(f"Existing model: {args.existing_model_path}")
    else:
        existing_model_path = None
    model = load_model(model_config_path, existing_model_path)
    initial_epoch = model.state_dict().get("epoch", 0)
    LOGGER.info(f"Initial epoch: {initial_epoch}")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    # DATASET
    LOGGER.info("dataset...")
    dataset_dir = pathlib.Path(args.dataset_dir)
    expected_image_size = model.expected_image_size
    classes = None  # ALL CLASSES
    train_dataset, val_dataset = load_datasets(dataset_dir, expected_image_size)
    for i in range(10):
        img_tensor, annotations = val_dataset.get_pair(i)

        # to opencv image
        print(img_tensor.min())
        print(img_tensor.max())
