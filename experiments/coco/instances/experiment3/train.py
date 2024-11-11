import argparse
import math
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
    train_dataset = mnn_ordinal.COCOInstances2017Ordinal2(
        dataset_dir,
        "train",
        expected_image_size,
    )
    val_dataset = mnn_ordinal.COCOInstances2017Ordinal2(
        dataset_dir,
        "val",
        expected_image_size,
    )
    return train_dataset, val_dataset


def load_model(
    config_path: pathlib.Path, existing_model_path: Optional[pathlib.Path] = None
) -> mnn_vit_model.VitObjectDetectionNetwork:
    model_config, _, head_config = mnn_vit_config.load_model_config(config_path)
    model = mnn_vit_model.VitObjectDetectionNetwork2(
        model_config=model_config,
        head_config=head_config,
        head_activation=torch.nn.Sigmoid(),
    )
    if existing_model_path:
        model.load_state_dict(torch.load(existing_model_path))
    return model


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
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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


def calculate_loss_n_objects(
    output0: torch.Tensor,
    target0: torch.Tensor,
    loss_fn: torch.nn.Module,
):
    # 'output0' is the number of objects
    ## We only care about the first vector from output0
    pred_n_objects = output0[:, 0, :]
    target_n_objects = target0[:, 0, :]
    loss1 = loss_fn(pred_n_objects, target_n_objects)
    ## The rest should all be zeros
    pred_n_objects_rest = output0[:, 1:, :]
    target_n_objects_rest = target0[:, 1:, :]
    loss1_rest = loss_fn(pred_n_objects_rest, target_n_objects_rest)
    total_loss0 = loss1 + loss1_rest
    return total_loss0


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    LOGGER.info(f"b1_x1: {b1_x1.shape}")
    LOGGER.info(f"b2_x1: {b2_x1.shape}")
    LOGGER.info(f"b1_x2: {b1_x2.shape}")
    LOGGER.info(f"b2_x2: {b2_x2.shape}")
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(
            b2_x1
        )  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def calculate_bbox_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_fn: torch.nn.Module,
    dataset: mnn_ordinal.COCOInstances2017Ordinal2,
):
    loss_bbox = 0
    loss_categories = 0
    for output_sample, target_sample in zip(output, target):
        # Prepare predictions
        pred_bboxes, pred_categories, _ = dataset.decode_prediction_raw(output_sample)
        pred_bboxes_tensor = torch.Tensor(pred_bboxes)
        pred_categories_tensor = torch.Tensor(pred_categories)

        # Prepare targets
        n_objects_vector = target_sample[0, 0, :]
        n_objects = min(torch.argmax(n_objects_vector), dataset.TOTAL_POSSIBLE_OBJECTS)
        target_bboxes_tensor = torch.zeros(pred_bboxes_tensor.shape)
        target_categories_tensor = torch.zeros(pred_categories_tensor.shape)
        if n_objects > 0:
            objects_dim = target_sample[1, :, :].shape[0]
            times_to_repeat = objects_dim // n_objects
            rest = objects_dim % n_objects
            target_bboxes, target_categories, _ = dataset.decode_output_tensor(
                target_sample
            )
            t_bboxes = target_bboxes.copy()
            t_categories = target_categories.copy()
            for _ in range(times_to_repeat):
                target_bboxes += t_bboxes
                target_categories += t_categories
                # target_objectnesses.extend(target_objectnesses)
            if rest > 0:
                target_bboxes += t_bboxes[:rest]
                target_categories += t_categories[:rest]
                # target_objectnesses.extend(target_objectnesses[:rest])

            target_bboxes_tensor = torch.Tensor(target_bboxes)
            target_categories_tensor = torch.Tensor(target_categories)
            LOGGER.info(target_bboxes_tensor.shape)

        ious = bbox_iou(pred_bboxes_tensor, target_bboxes_tensor)
        loss_iou = 1.0 - ious
        loss_bbox += loss_iou.sum()
        loss_categories += loss_fn(
            pred_categories_tensor.sigmoid(), target_categories_tensor.sigmoid()
        )  # TODO: Not sure about the sigmoid

    return loss_bbox / output.shape[0], loss_categories / output.shape[0]


def calculate_loss(
    output0: torch.Tensor,
    target0: torch.Tensor,
    output1: torch.Tensor,
    target1: torch.Tensor,
    loss_fn: torch.nn.Module,
):
    loss = []
    # 'output0' is the number of objects
    ## We only care about the first vector from output0
    pred_n_objects = output0[:, 0, :]
    target_n_objects = target0[:, 0, :]
    loss1 = loss_fn(pred_n_objects, target_n_objects)
    ## The rest should all be zeros
    pred_n_objects_rest = output0[:, 1:, :]
    target_n_objects_rest = target0[:, 1:, :]
    loss1_rest = loss_fn(pred_n_objects_rest, target_n_objects_rest)
    total_loss0 = loss1 + loss1_rest
    loss.append(total_loss0)

    # 'output1' is the bounding boxes
    ## Compute loss for each bounding box
    total_loss = 0
    for i in range(output1.shape[1]):
        pred_bboxes = output1[:, i, :]
        target_bboxes = target1[:, i, :]
        total_loss += loss_fn(pred_bboxes, target_bboxes)
    ## And then all bounding boxes together
    total_loss += loss_fn(output1, target1)
    return total_loss


def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    hyperparameters_config: mnn_config.HyperparametersConfiguration,
    current_epoch: int,
    validation_image_unsqueezed: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    writer: torch.utils.tensorboard.SummaryWriter = None,
    log_rate: int = 1000,
    model_save_path: pathlib.Path = pathlib.Path("my_vit_object_detection.pth"),
    scheduler=None,
) -> None:

    model.train()  # important for batch normalization and dropout layers
    running_loss = 0
    running_iou_05 = 0

    val_counter = 0
    tqdm_obj = tqdm.tqdm(train_loader, desc="Training | Loss: 0 | IoU: 0")
    for i, (image_batch, target) in enumerate(tqdm_obj):
        # Prepare input
        image_batch = image_batch.to(
            device=device,
            dtype=hyperparameters_config.floating_point_precision,
            non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
        )
        # Prepare outputs targets
        target0 = target[:, 0]  # Number of objects
        target1 = target[:, 1]  # Bounding boxes
        target0 = target0.to(
            device=device,
            dtype=hyperparameters_config.floating_point_precision,
            non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
        )
        target1 = target1.to(
            device=device,
            dtype=hyperparameters_config.floating_point_precision,
            non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
        )

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass
        output0, output1 = model(image_batch)

        # total_loss = calculate_loss(output0, target0, output1, target1, loss_fn)
        loss_n_object = calculate_loss_n_objects(output0, target0, loss_fn)
        loss_bbox, loss_categories = calculate_bbox_loss(
            torch.cat([output0.unsqueeze(1), output1.unsqueeze(1)], dim=1),
            target,
            loss_fn,
            train_loader.dataset,
        )
        total_loss = loss_n_object + loss_bbox + loss_categories
        total_loss.backward()

        if scheduler is not None:
            scheduler.add_batch_loss(total_loss)
        # Adjust learning weights
        optimizer.step()

        # Log metrics
        training_step = i + current_epoch * len(train_loader)
        current_loss = total_loss.item()
        tqdm_obj.set_description(
            f"Training | Total: {current_loss:.4f} | Bbox: {loss_bbox.item():.4f} | n-objects : {loss_n_object.item():.4f} | Categories: {loss_categories.item():.4f}"
        )
        if writer is not None:
            writer.add_scalar("Loss/train_total", current_loss, training_step)
            writer.add_scalar(
                "Loss/train_bbox",
                loss_bbox.item(),
                training_step,
            )
            writer.add_scalar(
                "Loss/train_categories",
                loss_bbox.item(),
                training_step,
            )
            writer.add_scalar(
                "Loss/train_n_objects",
                loss_n_object.item(),
                training_step,
            )

        running_loss += current_loss
        running_iou_05 += loss_bbox
        if i % log_rate == 0:
            model_state = model.state_dict()
            model_state["epoch"] = current_epoch
            model_state["step"] = training_step
            last_loss = running_loss / log_rate
            model_state["loss"] = running_loss / log_rate
            torch.save(model_state, model_save_path)

            running_loss = 0
            running_iou_05 = 0

            #### val image
            temp_out0, temp_out1 = model(validation_image_unsqueezed)
            temp_output = torch.stack(
                [temp_out0.squeeze(0), temp_out1.squeeze(0)], dim=0
            )
            pred_bboxes, pred_categories, pred_objectnesses = (
                train_loader.dataset.decode_output_tensor(temp_output)
            )
            write_image_with_output_of_experiment3(
                pred_bboxes,
                pred_categories,
                pred_objectnesses,
                validation_image_unsqueezed.squeeze(0),
                "validation_image_prediction",
            )
            #### train image
            target_bboxes, target_categories, target_objectnesses = (
                train_loader.dataset.decode_output_tensor(target[0])
            )
            write_image_with_output_of_experiment3(
                target_bboxes,
                target_categories,
                target_objectnesses,
                image_batch[0],
                "train_image_target",
            )
            p_bboxes, p_categories, p_objectnesses = (
                train_loader.dataset.decode_output_tensor(
                    torch.stack([output0[0], output1[0]], dim=0)
                )
            )
            write_image_with_output_of_experiment3(
                p_bboxes,
                p_categories,
                p_objectnesses,
                image_batch[0],
                "train_image_prediction",
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
        tqdm_obj = tqdm.tqdm(
            val_loader,
            desc=f"Validation | Total: 0 | Bbox: 0 | n-objects : 0 | Categories: 0",
        )
        for i, (image_batch, target) in enumerate(tqdm_obj):
            # Prepare input
            image_batch = image_batch.to(
                device=device,
                dtype=hyperparameters_config.floating_point_precision,
                non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
            )
            # Prepare outputs targets
            target0 = target[:, 0]  # Number of objects
            target1 = target[:, 1]  # Bounding boxes
            target0 = target0.to(
                device=device,
                dtype=hyperparameters_config.floating_point_precision,
                non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
            )
            target1 = target1.to(
                device=device,
                dtype=hyperparameters_config.floating_point_precision,
                non_blocking=True,  # Requires data loader use flag 'pin_memory=True'
            )

            # Forward pass
            output0, output1 = model(image_batch)

            loss_n_object = calculate_loss_n_objects(output0, target0, loss_fn)
            loss_bbox, loss_categories = calculate_bbox_loss(
                torch.cat([output0.unsqueeze(1), output1.unsqueeze(1)], dim=1),
                target,
                loss_fn,
                val_loader.dataset,
            )
            total_loss = loss_n_object + loss_bbox + loss_categories
            total_loss.backward()

            # Log metrics
            validation_step = i + current_epoch * len(val_loader)
            current_loss = total_loss.item()
            tqdm_obj.set_description(
                f"Validation | Total: {current_loss:.4f} | Bbox: {loss_bbox.item():.4f} | n-objects : {loss_n_object.item():.4f} | Categories: {loss_categories.item():.4f}"
            )
            if writer is not None:
                writer.add_scalar("Loss/val_total", current_loss, validation_step)
                writer.add_scalar(
                    "Loss/val_bbox",
                    loss_bbox.item(),
                    validation_step,
                )
                writer.add_scalar(
                    "Loss/val_categories",
                    loss_bbox.item(),
                    validation_step,
                )
                writer.add_scalar(
                    "Loss/val_n_objects",
                    loss_n_object.item(),
                    validation_step,
                )

            running_loss += current_loss
            if i % log_rate == 0 and i > 0:
                last_loss = running_loss / log_rate
        return last_loss


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
    log_rate: int = 50,
    save_dir: pathlib.Path = pathlib.Path("trained_models"),
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

    # Prepare validation image
    validation_image, target = val_dataset[2]
    validation_image_unsqueezed = validation_image.to(
        device, dtype=hyperparameters_config.floating_point_precision
    ).unsqueeze(0)
    target_bboxes, target_categories, target_confidences = (
        val_dataset.decode_output_tensor(target)
    )
    write_image_with_output_of_experiment3(
        target_bboxes,
        target_categories,
        target_confidences,
        validation_image_unsqueezed.squeeze(0),
        "validation_image_gt",
    )

    output0, output1 = object_detection_model(validation_image_unsqueezed)
    pred_bboxes, pred_categories, pred_confidences = train_dataset.decode_output_tensor(
        torch.stack([output0.squeeze(0), output1.squeeze(0)], dim=0)
    )
    write_image_with_output_of_experiment3(
        pred_bboxes,
        pred_categories,
        pred_confidences,
        validation_image_unsqueezed.squeeze(0),
        "pre-session_prediction",
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

    for epoch in range(hyperparameters_config.epochs):
        LOGGER.info(f"---------- EPOCH-{epoch} ------------")
        train_one_epoch(
            train_loader,
            object_detection_model,
            optimizer,
            loss_fn,
            hyperparameters_config,
            epoch,
            validation_image_unsqueezed=validation_image_unsqueezed,
            device=device,
            writer=writer,
            log_rate=log_rate,
            model_save_path=model_between_epoch_save_path,
            scheduler=scheduler,  # TODO MAKE TYPES RIGHT
        )
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

        model_state = object_detection_model.state_dict()
        model_state["epoch"] = epoch
        torch.save(model_state, model_save_path)


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

    parameters_grouped = get_params_grouped(model)
    lr = hyperparameters_config.learning_rate
    momentum = 0.937
    decay = 0.001
    optimizer = torch.optim.Adam(
        parameters_grouped[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
    )
    optimizer.add_param_group({"params": parameters_grouped[0], "weight_decay": decay})
    optimizer.add_param_group({"params": parameters_grouped[1], "weight_decay": 0.0})

    scheduler = MyLRScheduler(optimizer)

    # Tensorboard
    LOGGER.info("tensorboard summary writer. Open with: \ntensorboard --logdir=runs")
    writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=f"runs/coco_my_vit_predictions"
    )

    LOGGER.info("------ TRAIN & VAL ------")
    # Train & Validate session
    train_val(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        object_detection_model=model,
        loss_fn=FocalLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparameters_config=hyperparameters_config,
        writer=writer,
        experiment="experiment3",
    )
