import torch
import pathlib

import tqdm
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
    model_save_path: pathlib.Path = pathlib.Path("my_vit_object_detection.pth"),
) -> None:

    model.train()  # important for batch normalization and dropout layers
    running_loss = 0
    running_iou_05 = 0

    validation_image = mnn_train_utils.prepare_validation_image(
        validation_image_path, model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)

    val_counter = 0
    tqdm_obj = tqdm.tqdm(train_loader, desc="Training | Loss: 0 | IoU-0.5: 0")
    for i, (image_batch, target0) in enumerate(tqdm_obj):
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
            mnn_metrics.calculate_iou_batch(output, target0, threshold=0.5)
            .mean()
            .item()
        )
        tqdm_obj.set_description(
            f"Training | Loss: {current_loss:.4f} | IoU-0.5: {current_iou_05:.4f}"
        )
        if writer is not None:
            writer.add_scalar("Loss/train", current_loss, training_step)
            writer.add_scalar(
                "IoU_0.5/train",
                current_iou_05,
                training_step,
            )

        running_loss += current_loss
        running_iou_05 += current_iou_05
        if i % log_rate == 0:
            torch.save(model.state_dict(), model_save_path)
            with open("model_steps_till_now.txt", "w") as f:
                f.write(
                    f"Trained till step: {training_step} | Total steps per epoch: {len(train_loader)}"
                )
            last_loss = running_loss / log_rate
            print(
                f"Training step {training_step} | Loss: {last_loss:.4f} | IoU-0.5: {running_iou_05 / log_rate:.4f}",
            )
            running_loss = 0
            running_iou_05 = 0

            # Store validation image to inspect the model's performance
            temp_out = model(validation_image)
            mnn_train_utils.write_image_with_mask(
                temp_out, validation_image, "validation_image_prediction"
            )
            mnn_train_utils.write_image_with_mask(
                target0[0].unsqueeze(0),
                image_batch[0].unsqueeze(0),
                "train_image_ground_truth",
            )
            val_counter += 1
    return last_loss


def val_once(
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    current_epoch: int,
    device: torch.device = torch.device("cpu"),
    writer: torch.utils.tensorboard.SummaryWriter = None,
    log_rate: int = 1000,
):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        running_iou_05 = 0
        tqdm_obj = tqdm.tqdm(val_loader, desc="Validation | Loss: 0 | IoU-0.5: 0")
        for i, (image_batch, target0) in enumerate(tqdm_obj):
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

            output = model(image_batch)

            loss = loss_fn(output, target0)

            current_loss = loss.item()
            current_iou_05 = (
                mnn_metrics.calculate_iou_batch(output, target0, threshold=0.5)
                .mean()
                .item()
            )
            validation_step = i + current_epoch * len(val_loader)
            tqdm_obj.set_description(
                f"Validation | Loss: {current_loss:.4f} | IoU-0.5: {current_iou_05:.4f}"
            )
            if writer is not None:
                writer.add_scalar("Loss/val", current_loss, validation_step)
                writer.add_scalar(
                    "IoU_0.5/val",
                    current_iou_05,
                    i + current_epoch * len(val_loader),
                )

            running_loss += current_loss
            running_iou_05 += current_iou_05
            if i % log_rate == 0 and i > 0:
                last_loss = running_loss / log_rate
                print(
                    f"Validation step {validation_step}, loss: {last_loss:.4f} IoU-0.5: {running_iou_05 / log_rate:.4f}"
                )
        return last_loss
