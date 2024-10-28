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
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)


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
        dataset_dir,
        "train",
        expected_image_size,
        classes=classes,
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

    # HYPERPARAMETERS
    LOGGER.info("hyperparameters...")
    hyperparameters_config_path = pathlib.Path(args.hyperparameters_config_path)
    hyperparameters_config = load_hyperparameters_config(hyperparameters_config_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lrf = 0.01
    lf = (
        lambda x: max(1 - x / hyperparameters_config.epochs, 0) * (1.0 - lrf) + lrf
    )  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for i in range(hyperparameters_config.epochs):
        scheduler.step()

    # Tensorboard
    LOGGER.info("tesnorboard summary writer. Open with: \ntensorboard --logdir=runs")
    writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=f"runs/coco_my_vit_predictions"
    )

    # Validation Image
    validation_image_path = dataset_dir / "val2017" / "000000000139.jpg"

    LOGGER.info("------ TRAIN & VAL ------")
    # Train & Validate session
    train_val(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        object_detection_model=model,
        loss_fn=torch.nn.BCELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        scheduler=scheduler,
        hyperparameters_config=hyperparameters_config,
        writer=writer,
        experiment="experiment1",
        validation_image_path=validation_image_path,
        io_transform=None,
    )
