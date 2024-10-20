import torch
from torch.utils.tensorboard import SummaryWriter

import mnn.vision.image_size
import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.loader


from mnn.vision.dataset.coco.training.transform import BaseIOTransform
from mnn.vision.dataset.coco.training.utils import *
from mnn.vision.dataset.coco.training.session import train_one_epoch, val_once


def train_val(
    dataset_dir: pathlib.Path,
    object_detection_model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    validation_image_path: pathlib.Path,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    experiment: str,
    io_transform: BaseIOTransform = None,
    prediction_transform: BaseIOTransform = None,
    log_rate: int = 50,
    save_dir: pathlib.Path = pathlib.Path("trained_models"),
):

    print(
        f"Created model with {count_parameters(object_detection_model) / (10 ** 6)} million parameters"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    validation_image = prepare_validation_image(
        validation_image_path, object_detection_model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)
    temp_out = object_detection_model(validation_image)
    write_image_with_mask(temp_out, validation_image, "pre-session_prediction")

    # See coco["categories"]
    # classes = [1]
    classes = None
    train_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir,
        "train",
        object_detection_model.expected_image_size,
        classes=classes,
    )
    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", object_detection_model.expected_image_size, classes=classes
    )

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
    print("- Open tensorboard with:\ntensorboard --logdir=runs")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    model_save_path = save_dir / f"{experiment}_object_detection.pth"

    # Copied from ultralytics
    lrf = 0.01
    lf = (
        lambda x: max(1 - x / hyperparameters_config.epochs, 0) * (1.0 - lrf) + lrf
    )  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(hyperparameters_config.epochs):
        print(f"---------- EPOCH-{epoch} ------------")
        train_one_epoch(
            train_loader,
            object_detection_model,
            optimizer,
            loss_fn,
            hyperparameters_config,
            epoch,
            io_transform=io_transform,
            prediction_transform=prediction_transform,
            device=device,
            validation_image_path=validation_image_path,
            writer=writer,
            log_rate=log_rate,
            model_save_path=model_save_path,
        )
        torch.save(object_detection_model.state_dict(), model_save_path)
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
