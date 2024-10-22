import torch
from torch.utils.tensorboard import SummaryWriter

import mnn.vision.dataset.coco.training.train as coco_train
from mnn.vision.dataset.coco.training.transform import BaseIOTransform
import mnn.vision.image_size
from mnn.vision.models.vision_transformer.e2e import DoubleRGBCombinator
from mnn.vision.models.vision_transformer.tasks.object_detection import (
    ObjectDetectionOrdinalHead,
)
import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.loader


from mnn.vision.dataset.coco.training.utils import *


class VitObjectDetectionNetwork(torch.nn.Module):

    def __init__(
        self,
        model_config: mnn_encoder_config.MyBackboneVitConfiguration,
        head_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
    ):
        super().__init__()
        expected_image_width = model_config.rgb_combinator_config.d_model
        expected_image_height = (
            model_config.rgb_combinator_config.feed_forward_dimensions
        )
        self.expected_image_size = mnn.vision.image_size.ImageSize(
            width=expected_image_width, height=expected_image_height
        )
        self.encoder = DoubleRGBCombinator(model_config, self.expected_image_size)
        self.head = ObjectDetectionOrdinalHead(config=head_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.head(x)


class FocalLoss(torch.nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = "mean"

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.reduce == "sum":
            return torch.sum(loss)
        elif self.reduce == "mean":
            return torch.mean(loss)
        elif self.reduce == "none":
            return loss
        else:
            raise ValueError(
                "The value of the reduce parameter should be either 'sum', 'mean' or 'none'"
            )


class IOTransform(BaseIOTransform):
    LOWEST_FACTOR = 0.5
    HIGHEST_FACTOR = 1.0
    INCREMENT_FREQUENCY = 100  # in steps

    step_counter = 0
    scale_factor = 0.5
    scale_factor_incremental_step = 0.02

    def transform_input(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def transform_output(self, batch: torch.Tensor) -> torch.Tensor:
        if self.scale_factor < 1.0:
            batch = batch.unsqueeze(1)
            batch = torch.nn.functional.interpolate(
                batch,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
            batch = batch.squeeze(1)
        return batch

    def update_transform_configuration(self):
        self.step_counter += 1
        if self.step_counter % self.INCREMENT_FREQUENCY == 0:
            if not (self.LOWEST_FACTOR <= self.scale_factor <= self.HIGHEST_FACTOR):
                self.scale_factor_incremental_step *= -1
            self.scale_factor += self.scale_factor_incremental_step


if __name__ == "__main__":

    model_config, encoder_config, head_config = load_model_config(
        pathlib.Path("model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("hyperparameters.yaml")
    )
    validation_image_path = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/val2017/000000000139.jpg"
    )

    object_detection_model = VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )

    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/"
    )

    optimizer = torch.optim.AdamW(
        object_detection_model.parameters(), lr=hyperparameters_config.learning_rate
    )
    loss_fn = FocalLoss(gamma=0.75)

    # TensorBoard writer
    experiment = "exp10"
    writer = SummaryWriter(log_dir=f"runs/{experiment}_coco_my_vit_normed_predictions")
    print("- Open tensorboard with:\ntensorboard --logdir=runs")

    coco_train.train_val(
        dataset_dir=dataset_dir,
        object_detection_model=object_detection_model,
        loss_fn=loss_fn,
        validation_image_path=validation_image_path,
        hyperparameters_config=hyperparameters_config,
        optimizer=optimizer,
        writer=writer,
        experiment=experiment,
        io_transform=IOTransform(),
        prediction_transform=IOTransform(),
        log_rate=1000,
    )
