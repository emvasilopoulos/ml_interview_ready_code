import torch
from torch.utils.tensorboard import SummaryWriter

import mnn.vision.image_size
import mnn.vision.dataset.coco.training.train as coco_train
import mnn.vision.dataset.utilities
import mnn.vision.dataset.coco.torch_dataset
from mnn.vision.dataset.coco.training.transform import BaseIOTransform
from mnn.vision.models.vision_transformer.e2e import RGBCombinator
from mnn.vision.models.vision_transformer.components.encoder.vit_encoder import (
    RawVisionTransformerRGBEncoder,
)
import mnn.vision.models.heads.object_detection
import mnn.vision.models.vision_transformer.components.encoder.utils as mnn_encoder_utils


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

        combinator_activation = mnn_encoder_utils.get_combinator_activation_from_config(
            model_config.rgb_combinator_config
        )
        self.rgb_combinator = RGBCombinator(
            encoder=RawVisionTransformerRGBEncoder(
                model_config.rgb_combinator_config,
                self.expected_image_size,
            ),
            combinator_activation=combinator_activation,
        )
        self.hidden_transformer0 = (
            mnn_encoder_utils.get_transformer_encoder_from_config(
                model_config.encoder_config
            )
        )
        self.hidden_transformer1 = (
            mnn_encoder_utils.get_transformer_encoder_from_config(head_config)
        )

        layer_norm_eps = model_config.encoder_config.layer_norm_config.eps
        bias = model_config.encoder_config.layer_norm_config.bias
        self.layer_norm0 = torch.nn.LayerNorm(
            model_config.encoder_config.d_model, eps=layer_norm_eps, bias=bias
        )
        self.layer_norm1 = torch.nn.LayerNorm(
            model_config.encoder_config.d_model, eps=layer_norm_eps, bias=bias
        )

        self.layer_norm2 = torch.nn.LayerNorm(
            model_config.encoder_config.d_model, eps=layer_norm_eps, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean_ch = x.mean(dim=1)
        x0 = self.rgb_combinator(x)
        x_ = self.layer_norm0(x0 + x_mean_ch)  # Residual
        # print(f"Layer norm-0: {x_.min()}, {x_.max()}")

        x1 = self.hidden_transformer0(x0)
        x_ = self.layer_norm1(x1 + x_ + x_mean_ch)  # Residual
        # print(f"Layer norm-1: {x_.min()}, {x_.max()}")

        x2 = self.hidden_transformer1(x1)
        x_ = self.layer_norm2(x_mean_ch + x0 + x1 + x2)  # Residual
        # print(f"Layer norm-2: {x_.min()}, {x_.max()}")

        return x_  # possibly return x0, x1, x2


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
    loss_fn = torch.nn.MSELoss()

    # TensorBoard writer
    experiment = "exp13"
    writer = SummaryWriter(log_dir=f"runs/{experiment}_coco_my_vit_normed_predictions")
    print("- Open tensorboard with:\ntensorboard --logdir=runs")

    save_dir = pathlib.Path("trained_models")
    coco_train.train_val(
        dataset_dir=dataset_dir,
        object_detection_model=object_detection_model,
        loss_fn=loss_fn,
        validation_image_path=validation_image_path,
        hyperparameters_config=hyperparameters_config,
        optimizer=optimizer,
        writer=writer,
        experiment=experiment,
        io_transform=None,
        prediction_transform=None,
        log_rate=100,
        save_dir=save_dir,
    )
