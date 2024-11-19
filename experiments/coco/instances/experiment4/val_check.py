import argparse
import pathlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.tensorboard

import mnn.logging
from mnn.losses import FocalLoss
from mnn.lr_scheduler import MyLRScheduler
from mnn.training_tools.parameters import get_params_grouped
from mnn.vision.config import load_hyperparameters_config
import mnn.vision.dataset.coco.experiments.detection_ordinal as mnn_ordinal
from mnn.vision.dataset.coco.training.metrics import bbox_iou
from mnn.vision.dataset.coco.training.train import train_val
import mnn.vision.image_size
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.model as mnn_vit_model
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.config as mnn_vit_config

LOGGER = mnn.logging.get_logger(__name__)


def load_datasets(
    dataset_dir: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
    classes: Optional[List[int]] = None,
) -> mnn_ordinal.COCOInstances2017Ordinal3:

    val_dataset = mnn_ordinal.COCOInstances2017Ordinal3(
        dataset_dir,
        "val",
        expected_image_size,
    )
    return val_dataset


def load_model(
    config_path: pathlib.Path, existing_model_path: Optional[pathlib.Path] = None
) -> mnn_vit_model.VitObjectDetectionNetwork:
    model_config, _, head_config = mnn_vit_config.load_model_config(config_path)
    model = mnn_vit_model.VitObjectDetectionNetwork3(
        model_config=model_config,
        head_config=head_config,
        head_activation=torch.nn.Sigmoid(),
    )
    if existing_model_path:
        model.load_state_dict(torch.load(existing_model_path))
    return model


focal_loss = FocalLoss()


class BboxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        iou = bbox_iou(predictions, targets)
        loss = 1.0 - iou.mean()
        return loss


class ExperimentalLoss(torch.nn.Module):

    def __init__(self, dataset: mnn_ordinal.COCOInstances2017Ordinal3):
        self.dataset = dataset
        super().__init__()
        self.loss_for_raw = FocalLoss(gamma=3)
        self.loss_for_objects_positions_in_grid = FocalLoss(gamma=2)
        self.loss_for_n_objects = torch.nn.MSELoss()
        self.loss_for_bboxes = BboxLoss()
        self.losses = {
            "raw": 0,
            "objects_positions_in_grid": 0,
            "n_objects": 0,
            "bboxes": 0,
        }

    def _total_loss(self):
        return sum(self.losses.values())

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # prediction_objectnesses = predictions[:, :, -1]
        # target_objectnesses = predictions[:, :, -1]
        # self.objects_positions_in_grid_loss = focal_loss(
        #     prediction_objectnesses, target_objectnesses
        # )
        # self.losses["objects_positions_in_grid"] = self.objects_positions_in_grid_loss

        # prediction_n_objects = predictions[:, :, -1].sum(dim=1)
        # target_n_objects = targets[:, :, -1].sum(dim=1)
        # self.n_objects_loss = torch.nn.functional.mse_loss(
        #     prediction_n_objects, target_n_objects
        # )
        # self.losses["n_objects"] = self.n_objects_loss

        predictions_boxes = self.dataset.output_to_bboxes_as_mask_batch(predictions)
        targets_boxes = self.dataset.output_to_bboxes_as_mask_batch(targets)
        self.bboxes_loss = self.loss_for_bboxes(predictions_boxes, targets_boxes)
        self.losses["bboxes"] = self.bboxes_loss

        # Whole output loss
        # self.raw_loss = focal_loss(predictions, targets)
        # self.losses["raw"] = self.raw_loss

        return self._total_loss()

    def latest_loss_to_tqdm(self) -> str:
        return " | ".join([f"{k}: {v:.4f}" for k, v in self.losses.items()])


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
    args = parser.parse_args()

    LOGGER.info("------ LOADING ------")
    # MODEL
    model_config_path = pathlib.Path(args.model_config_path)
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
    val_dataset = load_datasets(dataset_dir, expected_image_size)

    loss = ExperimentalLoss(val_dataset)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_loader_iter = iter(val_loader)
    images, targets = next(val_loader_iter)

    images = images.to(device)
    targets = targets.to(device)

    pred = model(images)

    # pred_bboxes_as_mask = val_dataset.output_to_bboxes_as_mask_batch(pred)
    targets_bboxes_as_mask = val_dataset.output_to_bboxes_as_mask_batch(targets)


    for i, (image, target_bboxes) in enumerate(zip(
        images, targets_bboxes_as_mask
    )):
        img = image.cpu().numpy().transpose(1, 2, 0)
        image_target = (img * 255).astype(np.uint8)
        image_target = cv2.cvtColor(image_target, cv2.COLOR_RGB2BGR)
        for bbox in target_bboxes:
            x, y, w, h = bbox
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            if all([x, y, w, h]):
                cv2.rectangle(image_target, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(f"target-{i}.png", image_target)

        # image_pred = (img * 255).astype(np.uint8)
        # image_pred = cv2.cvtColor(image_pred, cv2.COLOR_RGB2BGR)
        # for bbox in pred_bboxes:
        #     x, y, w, h = bbox
        #     x = int(x)
        #     y = int(y)
        #     w = int(w)
        #     h = int(h)
        #     if all([x, y, w, h]):
        #         cv2.rectangle(image_pred, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imwrite(f"pred-{i}.png", image_pred)
