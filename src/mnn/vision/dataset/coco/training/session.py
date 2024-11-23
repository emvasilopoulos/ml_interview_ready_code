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
    device: torch.device = torch.device("cpu"),
    validation_image_path: pathlib.Path = None,
    writer: torch.utils.tensorboard.SummaryWriter = None,
    log_rate: int = 1000,
    model_save_path: pathlib.Path = pathlib.Path("my_vit_object_detection.pth"),
    scheduler=None,
) -> None:

    model.train()  # important for batch normalization and dropout layers
    running_loss = 0
    running_iou_05 = 0

    validation_image = mnn_train_utils.prepare_validation_image(
        validation_image_path, model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)

    val_counter = 0
    tqdm_obj = tqdm.tqdm(train_loader, desc="Training | Loss: 0")
    for i, (image_batch, target0) in enumerate(tqdm_obj):
        if i == 0:
            continue
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

        # Compute the loss and its gradients
        loss = loss_fn(output, target0)
        loss.backward()
        if scheduler is not None:
            scheduler.add_batch_loss(loss)
        # Adjust learning weights
        optimizer.step()

        # Log metrics
        training_step = i + current_epoch * len(train_loader)
        current_loss = loss.item()

        tqdm_obj.set_description(f"Training | Loss: {loss_fn.latest_loss_to_tqdm()}")
        if writer is not None:
            writer.add_scalar("Loss/train", current_loss, training_step)

        running_loss += current_loss
        if i % log_rate == 0:
            model_state = model.state_dict()
            model_state["epoch"] = current_epoch
            model_state["step"] = training_step
            last_loss = running_loss / log_rate
            model_state["loss"] = running_loss / log_rate
            model_state["IoU"] = running_iou_05 / log_rate
            torch.save(model_state, model_save_path)

            running_loss = 0
            running_iou_05 = 0

            # Store validation image to inspect the model's performance
            temp_out = model(validation_image)
            # train_loader.dataset.write_image_with_model_output(
            #     temp_out.squeeze(0),
            #     validation_image.squeeze(0),
            #     f"validation_image_prediction",
            # )
            train_loader.dataset.write_image_with_model_output(
                target0[0],
                image_batch[0],
                "train_image_ground_truth",
            )
            # train_loader.dataset.write_image_with_model_output(
            #     output[0],
            #     image_batch[0],
            #     "train_image_pred",
            # )
            val_counter += 1
            exit()
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
        tqdm_obj = tqdm.tqdm(val_loader, desc="Validation | Loss: 0")
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
            validation_step = i + current_epoch * len(val_loader)
            tqdm_obj.set_description(
                f"Validation | Loss: {loss_fn.latest_loss_to_tqdm()}"
            )
            if writer is not None:
                writer.add_scalar("Loss/val", current_loss, validation_step)

            running_loss += current_loss
            if i % log_rate == 0 and i > 0:
                last_loss = running_loss / log_rate
        return last_loss
