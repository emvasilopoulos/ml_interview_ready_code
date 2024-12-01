import argparse
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.tensorboard
import torchvision.ops

import mnn.logging
from mnn.losses import FocalLoss
import mnn.lr_scheduler
from mnn.training_tools.parameters import get_params_grouped
from mnn.vision.config import load_hyperparameters_config
import mnn.vision.dataset.coco.experiments.ordinal.detection_ordinal as mnn_ordinal
from mnn.vision.dataset.coco.training.train import train_val
import mnn.vision.image_size
import mnn.vision.models.cnn.object_detection as mnn_vit_model

LOGGER = mnn.logging.get_logger(__name__)


def load_datasets(
    dataset_dir: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
    output_shape: mnn.vision.image_size.ImageSize,
) -> Tuple[
    mnn_ordinal.COCOInstances2017Ordinal,
    mnn_ordinal.COCOInstances2017Ordinal,
]:
    train_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir, "train", expected_image_size, output_shape
    )
    val_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir, "val", expected_image_size, output_shape
    )
    return train_dataset, val_dataset


def load_model(
    config_path: pathlib.Path, existing_model_path: Optional[pathlib.Path] = None
) -> mnn_vit_model.Vanilla:
    image_size = mnn.vision.image_size.ImageSize(676, 676)
    model = mnn_vit_model.Vanilla(image_size)
    if existing_model_path:
        model.load_state_dict(torch.load(existing_model_path))
    else:
        # Initialize weights using Kaiming Initialization
        def kaiming_init_weights(m):
            if isinstance(m, torch.nn.Linear):  # Check if the layer is a Linear layer
                torch.nn.init.kaiming_normal_(
                    m.weight, nonlinearity="leaky_relu"
                )  # For ReLU activation
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)  # Initialize biases to zero

        # Apply Kaiming Initialization
        model.apply(kaiming_init_weights)
    return model


class ExperimentalLoss(torch.nn.Module):
    counter = 0

    def __init__(self, dataset: mnn_ordinal.COCOInstances2017Ordinal):
        self.dataset = dataset
        super().__init__()
        self.xc_loss = torch.nn.BCELoss()
        self.yc_loss = torch.nn.BCELoss()
        self.w_loss = torch.nn.BCELoss()
        self.h_loss = torch.nn.BCELoss()
        self.objectness_loss = torch.nn.BCELoss()
        self.class_loss = torch.nn.BCELoss()
        self._reset_losses()
        self.total_loss = 0

    def _reset_losses(self):
        self.losses = {
            "xc": 0,
            "yc": 0,
            "w": 0,
            "h": 0,
            "objectness": 0,
            "class": 0,
        }

    def _total_loss(self):
        return sum(self.losses.values())

    def _add_sample_losses(
        self,
        name: str,
        loss: torch.Tensor,
        condition: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        loss_value = loss(
            pred[condition].unsqueeze(0),
            target[condition].unsqueeze(0),
        )
        self.losses[name] += loss_value

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        self._reset_losses()
        for pred_sample, target_sample in zip(predictions, targets):
            (
                pred_xc_ordinals,
                pred_yc_ordinals,
                pred_w_ordinals,
                pred_h_ordinals,
                pred_objectness_scores,
                pred_class_scores,
            ) = self.dataset.split_output_to_vectors(pred_sample)

            (
                target_xc_ordinals,
                target_yc_ordinals,
                target_w_ordinals,
                target_h_ordinals,
                target_objectness_scores,
                target_class_scores,
            ) = self.dataset.split_output_to_vectors(target_sample)
            positive_condition = target_objectness_scores > 0
            if positive_condition.sum() > 0:
                # condition = target_objectness_scores > 0 # could match with ordinal objectness
                condition = target_objectness_scores == 1
            else:
                condition = target_objectness_scores == 0
            self._add_sample_losses(
                "xc",
                self.xc_loss,
                condition,
                pred_xc_ordinals,
                target_xc_ordinals,
            )
            self._add_sample_losses(
                "yc",
                self.yc_loss,
                condition,
                pred_yc_ordinals,
                target_yc_ordinals,
            )
            self._add_sample_losses(
                "w",
                self.w_loss,
                condition,
                pred_w_ordinals,
                target_w_ordinals,
            )
            self._add_sample_losses(
                "h",
                self.h_loss,
                condition,
                pred_h_ordinals,
                target_h_ordinals,
            )
            self._add_sample_losses(
                "class",
                self.class_loss,
                condition,
                pred_class_scores,
                target_class_scores,
            )
            objectness_loss_value = self.objectness_loss(
                pred_objectness_scores,
                target_objectness_scores,
            )

            self.losses["objectness"] += objectness_loss_value

            # # # bbox
            # bboxes_gt, _, _, _ = self.dataset.decode_output_tensor(
            #     target_sample, filter_by_objectness_score=True
            # )
            # bboxes_pred, _, objectnesses_pred, _ = (
            #     self.dataset.decode_output_tensor_with_priors(
            #         pred_sample, filter_by_objectness_score=True
            #     )
            # )

            # # Average objectnesses because why not
            # objectnesses /= self.dataset.n_priors
            # bbox_loss_value = self.bbox_loss(bboxes_pred, objectnesses_pred, bboxes_gt)
            # self.losses["bbox"] += bbox_loss_value
        batch_size = targets.shape[0]
        self._average_losses(batch_size=batch_size)
        self.total_loss = self._total_loss()
        return self.total_loss

    def _average_losses(self, batch_size: int):
        for k in self.losses.keys():
            self.losses[k] /= batch_size

    def latest_loss_to_tqdm(self) -> str:
        return f"Total: {self.total_loss:.3f} | " + " | ".join(
            [f"{k}: {v:.3f}" for k, v in self.losses.items()]
        )


class ExperimentalLoss2(torch.nn.Module):

    def __init__(self, dataset: mnn_ordinal.COCOInstances2017Ordinal):
        super().__init__()
        self.loss = torch.nn.BCELoss()
        self.current_loss = 0
        self.dataset = dataset

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.current_loss = self.loss(predictions, targets)

        return self.current_loss

    def latest_loss_to_tqdm(self) -> str:
        return f"BCE-loss: {self.current_loss:.5f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=False,
        default="/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco",
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
    if args.existing_model_path is not None:
        existing_model_path = pathlib.Path(args.existing_model_path)
        LOGGER.info(f"Existing model: {args.existing_model_path}")
    else:
        existing_model_path = None
    model = load_model(existing_model_path)
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
    train_dataset, val_dataset = load_datasets(
        dataset_dir, expected_image_size, model.output_shape
    )

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

    percentage = 0.1
    update_step_size = (
        percentage * len(train_dataset) // hyperparameters_config.batch_size
    )
    scheduler = mnn.lr_scheduler.StepLRScheduler(
        optimizer, update_step_size=update_step_size
    )

    # Tensorboard
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
        loss_fn=ExperimentalLoss(dataset=train_dataset),
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparameters_config=hyperparameters_config,
        writer=writer,
        experiment="experiment5",
        validation_image_path=validation_image_path,
    )
