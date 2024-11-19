import argparse
import pathlib
from typing import List, Optional, Tuple

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
) -> Tuple[
    mnn_ordinal.COCOInstances2017Ordinal4,
    mnn_ordinal.COCOInstances2017Ordinal4,
]:
    train_dataset = mnn_ordinal.COCOInstances2017Ordinal4(
        dataset_dir,
        "train",
        expected_image_size,
    )
    val_dataset = mnn_ordinal.COCOInstances2017Ordinal4(
        dataset_dir,
        "val",
        expected_image_size,
    )
    return train_dataset, val_dataset


def load_model(
    config_path: pathlib.Path, existing_model_path: Optional[pathlib.Path] = None
) -> mnn_vit_model.VitObjectDetectionNetwork:
    model_config, _, head_config = mnn_vit_config.load_model_config(config_path)
    model = mnn_vit_model.VitObjectDetectionNetwork4(
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
        iou = bbox_iou(predictions, targets).squeeze(-1)
        loss = 1.0 - iou.mean(dim=1)
        return loss.sum(dim=0)


class ExperimentalLoss(torch.nn.Module):
    counter = 0

    def __init__(self, dataset: mnn_ordinal.COCOInstances2017Ordinal4):
        self.dataset = dataset
        super().__init__()
        self.xc_loss = FocalLoss(gamma=2.5)
        self.yc_loss = FocalLoss(gamma=2.5)
        self.w_loss = FocalLoss(gamma=2.0)
        self.h_loss = FocalLoss(gamma=2.0)
        self.class_loss = FocalLoss(gamma=3.0)
        self.losses = {
            "xc": 0,
            "yc": 0,
            "w": 0,
            "h": 0,
            "class": 0,
        }
        self.total_loss = 0

    def _total_loss(self):
        return sum(self.losses.values())

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.losses = {
            "xc": 0,
            "yc": 0,
            "w": 0,
            "h": 0,
            "class": 0,
        }
        for pred_sample, target_sample in zip(predictions, targets):
            (
                pred_xc_ordinals,
                pred_yc_ordinals,
                pred_w_ordinals,
                pred_h_ordinals,
                pred_class_scores,
            ) = self.dataset.split_output_to_vectors(pred_sample)

            (
                target_xc_ordinals,
                target_yc_ordinals,
                target_w_ordinals,
                target_h_ordinals,
                target_class_scores,
            ) = self.dataset.split_output_to_vectors(target_sample)

            self.xc_loss_value = self.xc_loss(
                pred_xc_ordinals,
                target_xc_ordinals,
            )
            self.losses["xc"] += self.xc_loss_value

            self.yc_loss_value = self.yc_loss(
                pred_yc_ordinals,
                target_yc_ordinals,
            )
            self.losses["yc"] += self.yc_loss_value

            self.w_loss_value = self.w_loss(pred_w_ordinals, target_w_ordinals)
            self.losses["w"] += self.w_loss_value

            self.h_loss_value = self.h_loss(pred_h_ordinals, target_h_ordinals)
            self.losses["h"] += self.h_loss_value

            self.class_loss_value = self.class_loss(
                pred_class_scores,
                target_class_scores,
            )
            self.losses["class"] += self.class_loss_value

        self.total_loss = self._total_loss()
        return self.total_loss

    def latest_loss_to_tqdm(self) -> str:
        return f"Total: {self.total_loss:.5f} | " + " | ".join(
            [f"{k}: {v:.5f}" for k, v in self.losses.items()]
        )


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
    validation_image_path = dataset_dir / "val2017" / "000000000139.jpg"
    # Train & Validate session
    train_val(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        object_detection_model=model,
        loss_fn=ExperimentalLoss(val_dataset),
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparameters_config=hyperparameters_config,
        writer=writer,
        experiment="experiment4",
        validation_image_path=validation_image_path,
    )
