import argparse
import pathlib
from typing import List, Optional, Tuple
import numpy as np

import torch
import torch.utils.tensorboard

from mnn.vision.config import load_hyperparameters_config
import mnn.vision.image_size
from mnn.vision.dataset.coco.training.train import (
    train_val,
)
import mnn.vision.dataset.coco.experiments.detection_ordinal as mnn_ordinal
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.model as mnn_vit_model
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.config as mnn_vit_config
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)


def load_datasets(
    dataset_dir: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
    classes: Optional[List[int]] = None,
) -> Tuple[
    mnn_ordinal.COCOInstances2017Ordinal,
    mnn_ordinal.COCOInstances2017Ordinal,
]:
    train_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir,
        "train",
        expected_image_size,
    )
    val_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir, "val", expected_image_size
    )
    return train_dataset, val_dataset


def load_model(
    config_path: pathlib.Path, existing_model_path: Optional[pathlib.Path] = None
) -> mnn_vit_model.VitObjectDetectionNetwork:
    model_config, _, head_config = mnn_vit_config.load_model_config(config_path)
    model = mnn_vit_model.VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config, head_activation=torch.nn.Sigmoid()
    )
    if existing_model_path:
        model.load_state_dict(torch.load(existing_model_path))
    return model

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, inputs, targets):
        p = inputs
        ce_loss = self.bce_loss(inputs, targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum()

# Copied from ultralytics
def get_params_grouped(model: torch.nn.Module):
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    parameters_grouped = [[], [], []]
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:  # bias (no decay)
                parameters_grouped[2].append(param)
            elif isinstance(module, bn):  # weight (no decay)
                parameters_grouped[1].append(param)
            else:  # weight (with decay)
                parameters_grouped[0].append(param)
    return parameters_grouped

import collections
class MyLRScheduler:

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.param_groups_initial_lrs = []
        for param_group in self.optimizer.param_groups:
            self.param_groups_initial_lrs.append(param_group["lr"])

        self.losses = collections.deque(maxlen=50)
        self.loss_moving_average = collections.deque(maxlen=50)
        self.logger = mnn.logging.get_logger("MyLRScheduler")

    def _reset_moving_average(self):
        self.loss_moving_average = collections.deque(maxlen=50)

    def _fit_line(self, data_points: List[float]):
        x = [i for i in range(len(data_points))]
        y = data_points
        m, b = np.polyfit(x, y, 1)
        return m, b

    def add_batch_loss(self, loss: torch.nn.Module):
        self.losses.append(loss.item())
        current_mean = torch.Tensor(self.losses).mean()
        self.loss_moving_average.append(current_mean)
        if len(self.loss_moving_average) == self.loss_moving_average.maxlen:
            line_angle, b = self._fit_line(self.loss_moving_average)
            if line_angle > 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    temp = param_group["lr"]
                    param_group["lr"] *= 0.9
                    if param_group["lr"] < 0.000001:
                        param_group["lr"] = self.param_groups_initial_lrs[i]
                    self.logger.info(f"Updating 'lr' for param_group-{i} from '{temp:.6f}' to {param_group['lr']} ")
            self._reset_moving_average()

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
        raise NotImplementedError("Continueing training not supported yet. Something to do with scheduler")
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
    optimizer = torch.optim.Adam(parameters_grouped[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": parameters_grouped[0], "weight_decay": decay})
    optimizer.add_param_group({"params": parameters_grouped[1], "weight_decay": 0.0})

    scheduler = MyLRScheduler(optimizer)

    # Tensorboard
    LOGGER.info("tesnorboard summary writer. Open with: \ntensorboard --logdir=runs")
    writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=f"runs/coco_my_vit_predictions"
    )

    # Validation Image
    validation_image_path = dataset_dir / "val2017" / "000000000139.jpg"

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
        experiment="experiment2",
        validation_image_path=validation_image_path,
        io_transform=None,
    )
