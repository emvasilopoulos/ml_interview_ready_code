import torch

import mnn.vision.image_size
import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.torch_dataset
from mnn.vision.dataset.coco.training.utils import *

import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils
from mnn.vision.models.vision_transformer.encoder.vit_encoder import (
    RawVisionTransformerRGBEncoder,
)
from mnn.vision.models.vision_transformer.e2e import RGBCombinator


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

    device = torch.device("cuda:0")
    object_detection_model = VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )
    object_detection_model.to(device=device)
    object_detection_model.load_state_dict(
        torch.load("trained_models/exp12_object_detection.pth")
    )

    validation_image = prepare_validation_image(
        validation_image_path, object_detection_model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)

    with torch.no_grad():
        temp_out = object_detection_model(validation_image)
        temp_out = torch.sigmoid(temp_out)
        d = 20
        for i in range(1, d):
            perc = i / d
            threshold = int(perc * 255)
            # temp_out = torch.sigmoid(temp_out)
            write_image_with_mask(
                temp_out,
                validation_image,
                f"test_mask_threshold.{threshold}",
                threshold,
            )
