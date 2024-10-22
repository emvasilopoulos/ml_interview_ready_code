import torch
from torch.utils.tensorboard import SummaryWriter

import mnn.vision.image_size
import mnn.vision.dataset.coco.training.train as coco_train
import mnn.vision.dataset.utilities
import mnn.vision.dataset.coco.torch_dataset
from mnn.vision.models.vision_transformer.e2e import RGBCombinator
from mnn.vision.models.vision_transformer.encoder.vit_encoder import (
    RawVisionTransformerRGBEncoder,
)
import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils


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

        layer_norm_eps = model_config.rgb_combinator_config.layer_norm_config.eps
        bias = model_config.rgb_combinator_config.layer_norm_config.bias
        self.layer_norm0 = torch.nn.LayerNorm(
            model_config.rgb_combinator_config.d_model, eps=layer_norm_eps, bias=bias
        )
        self.layer_norm1 = torch.nn.LayerNorm(
            model_config.rgb_combinator_config.d_model, eps=layer_norm_eps, bias=bias
        )

        self.layer_norm2 = torch.nn.LayerNorm(
            model_config.rgb_combinator_config.d_model, eps=layer_norm_eps, bias=bias
        )

        self.head_activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean_ch = x.mean(dim=1)
        x0 = self.rgb_combinator(x)
        x_ = self.layer_norm0(x0 + x_mean_ch)  # Residual

        x1 = self.hidden_transformer0(x_.permute(0, 2, 1))
        x1 = x1.permute(0, 2, 1)
        x_ = self.layer_norm1(x_ + x0 + x1)  # Residual

        x2 = self.hidden_transformer1(x_)
        x_ = self.layer_norm2(x_ + x0 + x1 + x2)  # Residual

        return self.head_activation(x_)


if __name__ == "__main__":

    model_config, encoder_config, head_config = load_model_config(
        pathlib.Path("model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("hyperparameters.yaml")
    )
    validation_image_path = pathlib.Path(
        "/home/manos/ml_interview_ready_code/data/val2017/000000000139.jpg"
    )

    object_detection_model = VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )
    object_detection_model.to(
        device=torch.device("cuda:0"),
        dtype=hyperparameters_config.floating_point_precision,
    )

    dataset_dir = pathlib.Path("/home/manos/ml_interview_ready_code/data/")

    # Copied from ultralytics
    momentum = 0.937
    optimizer = torch.optim.Adam(
        object_detection_model.parameters(),
        lr=hyperparameters_config.learning_rate,
        betas=(momentum, 0.999),
        weight_decay=0,  # TODO - used for overfitting
    )
    loss_fn = torch.nn.BCELoss()

    # TensorBoard writer
    experiment = "exp16"
    writer = SummaryWriter(log_dir=f"runs/{experiment}_coco_my_vit_normed_predictions")
    print("- Open tensorboard with:\ntensorboard --logdir=runs")

    save_dir = pathlib.Path("trained_models")

    """
    NO TRANSFORMS THIS TIME
    """
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
