import pathlib
import logging
from typing import Tuple

import torch
import torch.utils.tensorboard


from mnn.vision.dataset.coco.experiments.detection_fading_bboxes_in_mask import (
    COCOInstances2017FBM,
)
from mnn.vision.dataset.coco.training.transform import BaseIOTransform
import mnn.vision.config as mnn_config
import mnn.vision.dataset.coco.training.utils as mnn_coco_training_utils
import mnn.torch_utils as mnn_utils
from mnn.vision.dataset.coco.training.session import train_one_epoch, val_once
from mnn.vision.image_size import ImageSize


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def default_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    # Copied from ultralytics
    lrf = 0.01
    lf = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf  # linear
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


def default_train_val_datasets(
    dataset_dir: pathlib.Path, expected_image_size: ImageSize
) -> Tuple[COCOInstances2017FBM, COCOInstances2017FBM]:
    # See coco["categories"]
    classes = None
    train_dataset = COCOInstances2017FBM(
        dataset_dir,
        "train",
        expected_image_size,
        classes=classes,
    )
    val_dataset = COCOInstances2017FBM(
        dataset_dir, "val", expected_image_size, classes=classes
    )
    return train_dataset, val_dataset


def train_val(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    object_detection_model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    writer: torch.utils.tensorboard.SummaryWriter,
    experiment: str,
    validation_image_path: pathlib.Path,
    io_transform: BaseIOTransform = None,
    log_rate: int = 50,
    save_dir: pathlib.Path = pathlib.Path("trained_models"),
):
    # Print model parameters
    LOGGER.info(
        f"Created model with {mnn_utils.count_parameters(object_detection_model) / (10 ** 6)} million parameters"
    )

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Prepare validation image
    validation_image = mnn_coco_training_utils.prepare_validation_image(
        validation_image_path, object_detection_model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)
    temp_out = object_detection_model(validation_image)
    mnn_coco_training_utils.write_image_with_mask(
        temp_out, validation_image, "pre-session_prediction"
    )

    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparameters_config.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hyperparameters_config.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # TensorBoard writer
    LOGGER.info("========|> Open tensorboard with:\ntensorboard --logdir=runs")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    model_save_path = save_dir / f"{experiment}_object_detection_full_epoch.pth"
    model_between_epoch_save_path = (
        save_dir / f"{experiment}_object_detection_in_epoch.pth"
    )

    for epoch in range(hyperparameters_config.epochs):
        LOGGER.info(f"---------- EPOCH-{epoch} ------------")
        train_one_epoch(
            train_loader,
            object_detection_model,
            optimizer,
            loss_fn,
            hyperparameters_config,
            epoch,
            io_transform=io_transform,
            device=device,
            validation_image_path=validation_image_path,
            writer=writer,
            log_rate=log_rate,
            model_save_path=model_between_epoch_save_path,
        )

        model_state = object_detection_model.state_dict()
        model_state["epoch"] = epoch
        torch.save(model_state, model_save_path)
        val_loss = val_once(
            val_loader,
            object_detection_model,
            loss_fn,
            hyperparameters_config,
            epoch,
            device=device,
            writer=writer,
            log_rate=log_rate,
        )
        scheduler.step(val_loss)
