import torch
from torch.utils.tensorboard import SummaryWriter

import mnn.vision.image_size
import mnn.vision.dataset.coco.training.train as coco_train
import mnn.vision.dataset.utilities
import mnn.vision.dataset.coco.torch_dataset
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
    object_detection_model.load_state_dict(
        torch.load("trained_models/exp12_object_detection_12epochs.pth")
    )
    """
    FAIL 
    Freeze RGB combinator | LR = 0.0001
    applied after the following steps:
    1. new model trained for 10 epochs with output scaling transform (see experiment12 IOTransform class) | LR = 0.001
    2. then trained for 2 epochs without the output scaling transform | LR = 0.0005
    
    object_detection_model.rgb_combinator.requires_grad_(False)
    """

    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/"
    )

    optimizer = torch.optim.AdamW(
        object_detection_model.parameters(), lr=hyperparameters_config.learning_rate
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # TensorBoard writer
    experiment = "exp12"
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
        log_rate=1000,
        save_dir=save_dir,
    )
