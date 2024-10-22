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
    device = torch.device("cuda:0")
    object_detection_model = VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config
    )
    object_detection_model.load_state_dict(
        torch.load("trained_models/exp16_object_detection_9epochs.pth")
    )
    object_detection_model.to(
        device=device, dtype=hyperparameters_config.floating_point_precision
    )

    dataset_dir = pathlib.Path("/home/manos/ml_interview_ready_code/data/")
    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", object_detection_model.expected_image_size, classes=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
    )

    object_detection_model.eval()
    loss_fn = torch.nn.BCELoss()

    os.makedirs("raw_validation", exist_ok=True)
    with torch.no_grad():
        for i in range(len(val_dataset)):
            image_batch, target0 = val_dataset[i]
            image_batch = image_batch.unsqueeze(0)
            target0 = target0.unsqueeze(0)

            image_name = val_dataset.images[i]["file_name"]

            image_batch = image_batch.to(
                device=device,
                dtype=hyperparameters_config.floating_point_precision,
            )
            target0 = target0.to(
                device=device,
                dtype=hyperparameters_config.floating_point_precision,
            )

            output = object_detection_model(image_batch)

            loss = loss_fn(output, target0)
            current_loss = loss.item()

            with open(f"raw_validation/{image_name}.txt", "w") as f:
                f.write(f"Loss: {current_loss}")

            write_image_with_mask(
                output[0], image_batch[0], f"raw_validation/{image_name}"
            )
            write_image_with_mask(
                target0[0], image_batch[0], f"raw_validation/{image_name}_gt"
            )

            if i == 5:
                break
