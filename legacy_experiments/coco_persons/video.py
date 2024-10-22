import cv2

import torch

import mnn.vision.image_size
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

        self.head_activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean_ch = x.mean(dim=1)
        x0 = self.rgb_combinator(x)
        x_ = self.layer_norm0(x0 + x_mean_ch)  # Residual

        x1 = self.hidden_transformer0(x_)
        x_ = self.layer_norm1(x_ + x0 + x1)  # Residual

        x2 = self.hidden_transformer1(x_)
        x_ = self.layer_norm2(x_ + x0 + x1 + x2)  # Residual

        return self.head_activation(x_)


if __name__ == "__main__":
    model_config, encoder_config, head_config = load_model_config(
        pathlib.Path("experiment15/model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("experiment15/hyperparameters.yaml")
    )
    object_detection_model = VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )
    object_detection_model.load_state_dict(
        torch.load("experiment15/trained_models/exp15_object_detection.pth")
    )
    object_detection_model.to(device=torch.device("cuda:0"))
    object_detection_model.eval()

    preprocessor = (
        mnn.vision.dataset.object_detection.fading_bboxes_in_mask.FadingBboxMasks
    )

    cap = cv2.VideoCapture(0)
    with torch.no_grad():
        while True:
            ret, img = cap.read()
            img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            padding_percent = random.random()
            img_tensor = preprocessor.cv2_image_to_tensor(img_)
            img_tensor = (
                preprocessor.preprocess_image(
                    img_tensor,
                    object_detection_model.expected_image_size,
                    padding_percent=padding_percent,
                )
                .unsqueeze(0)
                .to(device=torch.device("cuda:0"))
            )

            out = object_detection_model(img_tensor)
            out = out.squeeze(0).cpu().numpy()
            raw_mask = (out * 255).astype("uint8")
            mask = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)

            img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

            masked_image = cv2.addWeighted(img, 1, mask, 1, 0)

            cv2.imshow("Masked Image", masked_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
