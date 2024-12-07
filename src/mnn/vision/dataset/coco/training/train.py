import pathlib
import logging
from typing import Tuple

import torch
import torch.utils.tensorboard


from mnn.vision.dataset.coco.base import BaseCOCODatasetGrouped
from mnn.vision.dataset.coco.training.transform import BaseIOTransform
import mnn.vision.config as mnn_config
import mnn.torch_utils as mnn_utils
from mnn.vision.dataset.coco.training.session import train_one_epoch, val_once
from mnn.vision.image_size import ImageSize
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)


def train_val(
    train_dataset: BaseCOCODatasetGrouped,
    val_dataset: BaseCOCODatasetGrouped,
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
    start_epoch: int = 0,
):
    # Print model parameters
    LOGGER.info(
        f"Created model with {mnn_utils.count_parameters(object_detection_model) / (10 ** 6):.2f} million parameters"
    )

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

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
        shuffle=False,
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

    for epoch in range(start_epoch, hyperparameters_config.epochs):
        LOGGER.info(f"---------- EPOCH-{epoch} ------------")
        # LOGGER.info(f"Scheduler State:\n{scheduler.state_dict()}")
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
            scheduler=scheduler,  # TODO MAKE TYPES RIGHT
        )

        model_state = object_detection_model.state_dict()
        model_state["epoch"] = epoch
        torch.save(model_state, model_save_path)
        val_once(
            val_loader,
            object_detection_model,
            loss_fn,
            hyperparameters_config,
            epoch,
            device=device,
            writer=writer,
            log_rate=log_rate,
        )
