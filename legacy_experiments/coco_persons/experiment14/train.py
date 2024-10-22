from typing import Callable
import torch
from torch.utils.tensorboard import SummaryWriter

import mnn.vision.dataset.coco.training.transform as coco_train_transform
import mnn.vision.image_size
import mnn.vision.dataset.coco.training.train as coco_train
import mnn.vision.dataset.utilities
import mnn.vision.dataset.coco.torch_dataset
from mnn.vision.models.vision_transformer.e2e import RGBCombinator
from mnn.vision.models.vision_transformer.encoder.vit_encoder import (
    RawVisionTransformerRGBEncoder,
)
import mnn.vision.models.heads.object_detection as mnn_object_detection
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


class IOTransform(coco_train_transform.BaseIOTransform):
    split_perc: float = random.randint(30, 70) / 100
    is_horizontal: bool = random.randint(0, 1)
    not_do_mosaic: bool = random.randint(0, 1)

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        if self.batch_size % 2 != 0:
            raise ValueError("Batch size must be even")

        self.split_perc = self._random_perc()
        self.is_horizontal = self._random_horizontal()

    def _random_perc(self):
        return random.randint(30, 70) / 100

    def _random_horizontal(self):
        return random.randint(0, 1)

    def _random_do_mosaic(self):
        return random.randint(0, 1)

    def _glue_horizontal(
        self, sample_i0: torch.Tensor, sample_i1: torch.Tensor, split: int
    ):
        return torch.cat([sample_i0[..., :, :split], sample_i1[..., :, split:]], dim=-1)

    def _glue_vertical(
        self, sample_i0: torch.Tensor, sample_i1: torch.Tensor, split: int
    ):
        return torch.cat([sample_i0[..., :split, :], sample_i1[..., split:, :]], dim=-2)

    def _create_mosaic_of_two_samples(
        self,
        batch: torch.Tensor,
        split_perc: float,
        is_horizontal: bool,
        _glue_horiz_callable: Callable,
        _glue_vert_callable: Callable,
    ):
        h, w = batch.shape[-2:]
        new_batch = []
        for i in range(0, batch.shape[0], 2):
            sample_i0 = batch[i]
            sample_i1 = batch[i + 1]
            # Create mosaic image
            if is_horizontal:
                split = int(w * split_perc)
                new_sample = _glue_horiz_callable(sample_i0, sample_i1, split)
            else:
                split = int(h * split_perc)
                new_sample = _glue_vert_callable(sample_i0, sample_i1, split)
            new_batch.append(new_sample)
        return torch.stack(new_batch)

    def transform_input(self, batch: torch.Tensor) -> torch.Tensor:
        """
        images have shape (batch_size, channels, height, width)
        """
        if batch.shape[0] % 2 != 0 or self.not_do_mosaic:
            return batch
        return self._create_mosaic_of_two_samples(
            batch,
            self.split_perc,
            self.is_horizontal,
            self._glue_horizontal,
            self._glue_vertical,
        )

    def transform_output(self, batch: torch.Tensor) -> torch.Tensor:
        """
        masks have shape (batch_size, height, width)
        each mask holds the bounding boxes of the objects + a bbox for the whole image.
        The glue functions search for the bboxes that surround the whole image. Because the masks are ready,
        if we split them on a certain position, some sides of the bboxes after that position will be missing.
        So at the position of splitting the images we refill with values as if that was the whole image.
        """
        if batch.shape[0] % 2 != 0 or self.not_do_mosaic:
            return batch
        return self._create_mosaic_of_two_samples(
            batch,
            self.split_perc,
            self.is_horizontal,
            self._glue_horizontal,
            self._glue_vertical,
        )

    def update_transform_configuration(self) -> None:
        self.split_perc = self._random_perc()
        self.is_horizontal = self._random_horizontal()
        self.not_do_mosaic = random.randint(0, 1)


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

    # Copied from YOLOv5
    momentum = 0.9
    optimizer = torch.optim.AdamW(
        object_detection_model.parameters(),
        lr=hyperparameters_config.learning_rate,
        betas=(momentum, 0.999),
        weight_decay=0.0,
    )
    loss_fn = torch.nn.BCELoss()

    # TensorBoard writer
    experiment = "exp14"
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
        io_transform=IOTransform(batch_size=hyperparameters_config.batch_size),
        prediction_transform=None,
        log_rate=100,
        save_dir=save_dir,
    )
