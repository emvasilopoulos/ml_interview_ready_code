import torch
import pathlib

import torch
import torch.utils.tensorboard

import mnn.vision.config as mnn_config
import mnn.vision.dataset.coco.training.metrics as mnn_metrics
import mnn.vision.dataset.coco.training.transform as mnn_train_transform
import mnn.vision.dataset.coco.training.utils as mnn_train_utils


def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    current_epoch: int,
    io_transform: mnn_train_transform.BaseIOTransform = None,
    prediction_transform: mnn_train_transform.BaseIOTransform = None,
    device: torch.device = torch.device("cpu"),
    validation_image_path: pathlib.Path = None,
    writer: torch.utils.tensorboard.SummaryWriter = None,
    log_rate: int = 1000,
    model_name: str = "my_vit_object_detection.pth",
) -> None:
    """
    RTX A2000 - i5 7th Gen - 8GB RAM
    forward pass Time taken: 0.4396486282348633 seconds
    backward propagation Time taken: 0.9111251831054688 seconds
    writer Time taken: 0.02442193031311035 seconds
    loss Time taken: 0.0001838207244873047 seconds
    iou Time taken: 0.0010769367218017578 seconds
    """

    model.train()  # important for batch normalization and dropout layers
    running_loss = 0
    running_iou_025 = 0
    running_iou_05 = 0
    running_iou_075 = 0

    validation_image = mnn_train_utils.prepare_validation_image(
        validation_image_path, model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)

    val_counter = 0
    for i, (image_batch, target0) in enumerate(train_loader):
        image_batch = image_batch.to(
            device=device,
            dtype=hyperparameters_config.floating_point_precision,
            non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
        )
        target0 = target0.to(
            device=device,
            dtype=hyperparameters_config.floating_point_precision,
            non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
        )

        if io_transform is not None:
            image_batch = io_transform.transform_input(image_batch)
            target0 = io_transform.transform_output(target0)
            io_transform.update_transform_configuration()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass
        output = model(image_batch)
        if prediction_transform is not None:
            output = prediction_transform.transform_output(output)
            prediction_transform.update_transform_configuration()

        # Compute the loss and its gradients
        loss = loss_fn(output, target0)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Log metrics
        training_step = i + current_epoch * len(train_loader)
        current_loss = loss.item()
        current_iou_05 = (
            mnn_metrics.calculate_iou_batch(output, target0, threshold=0.25)
            .mean()
            .item()
        )
        current_iou_025 = (
            mnn_metrics.calculate_iou_batch(output, target0, threshold=0.5)
            .mean()
            .item()
        )
        current_iou_075 = (
            mnn_metrics.calculate_iou_batch(output, target0, threshold=0.75)
            .mean()
            .item()
        )
        if writer is not None:
            writer.add_scalar(
                "IoU_0.25/train",
                current_iou_025,
                training_step,
            )
            writer.add_scalar(
                "IoU_0.5/train",
                current_iou_05,
                training_step,
            )
            writer.add_scalar(
                "IoU_0.75/train",
                current_iou_075,
                training_step,
            )
            writer.add_scalar("Loss/train", current_loss, training_step)

        running_loss += current_loss
        running_iou_025 += current_iou_025
        running_iou_05 += current_iou_05
        running_iou_075 += current_iou_075
        if i % log_rate == 0:
            torch.save(model.state_dict(), model_name)
            with open("model_steps_till_now.txt", "w") as f:
                f.write(
                    f"Trained till step: {training_step} | Total steps per epoch: {len(train_loader)}"
                )

            print(
                f"Training step {training_step} | Loss: {running_loss / log_rate:.4f} | IoU-0.25: {running_iou_025 / log_rate:.4f} | IoU-0.5: {running_iou_05 / log_rate:.4f} | IoU-0.75: {running_iou_075 / log_rate:.4f}",
            )
            running_loss = 0
            running_iou_025 = 0
            running_iou_05 = 0
            running_iou_075 = 0

            # Store validation image to inspect the model's performance
            temp_out = model(validation_image)
            mnn_train_utils.write_validation_image_with_predicted_mask(
                temp_out, validation_image, f"epoch_{current_epoch}_id_{val_counter}"
            )
            val_counter += 1


def val_once(
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    current_epoch: int,
    io_transform: mnn_train_transform.BaseIOTransform = None,
    prediction_transform: mnn_train_transform.BaseIOTransform = None,
    device: torch.device = torch.device("cpu"),
    writer: torch.utils.tensorboard.SummaryWriter = None,
    log_rate: int = 1000,
):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        running_iou_025 = 0
        running_iou_05 = 0
        running_iou_075 = 0
        for i, (image_batch, target0) in enumerate(val_loader):
            image_batch = image_batch.to(
                device=device,
                dtype=hyperparameters_config.floating_point_precision,
                non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
            )
            target0 = target0.to(
                device=device,
                dtype=hyperparameters_config.floating_point_precision,
                non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
            )

            if io_transform is not None:
                image_batch = io_transform.transform_input(image_batch)
                target0 = io_transform.transform_output(target0)
                io_transform.update_transform_configuration()

            output = model(image_batch)
            if prediction_transform is not None:
                output = prediction_transform.transform_output(output)
                prediction_transform.update_transform_configuration()

            loss = loss_fn(output, target0)

            current_loss = loss.item()
            current_iou_025 = (
                mnn_metrics.calculate_iou_batch(output, target0, threshold=0.25)
                .mean()
                .item()
            )
            current_iou_05 = (
                mnn_metrics.calculate_iou_batch(output, target0, threshold=0.5)
                .mean()
                .item()
            )
            current_iou_075 = (
                mnn_metrics.calculate_iou_batch(output, target0, threshold=0.75)
                .mean()
                .item()
            )
            validation_step = i + current_epoch * len(val_loader)
            if writer is not None:
                writer.add_scalar(
                    "IoU_0.25/val",
                    current_iou_025,
                    i + current_epoch * len(val_loader),
                )
                writer.add_scalar(
                    "IoU_0.5/val",
                    current_iou_05,
                    i + current_epoch * len(val_loader),
                )
                writer.add_scalar(
                    "IoU_0.75/val",
                    current_iou_075,
                    i + current_epoch * len(val_loader),
                )
                writer.add_scalar("Loss/val", current_loss, validation_step)

            running_loss += current_loss
            running_iou_025 += current_iou_025
            running_iou_05 += current_iou_05
            running_iou_075 += current_iou_075
            if i % log_rate == 0:
                print(
                    f"Validation step {validation_step}, loss: {running_loss / log_rate:.4f} IoU-0.25: {running_iou_025 / log_rate:.4f} IoU-0.5: {running_iou_05 / log_rate:.4f} IoU-0.7: {running_iou_075 / log_rate:.4f}"
                )
                running_loss = 0
                running_iou_025 = 0
                running_iou_05 = 0
                running_iou_075 = 0
