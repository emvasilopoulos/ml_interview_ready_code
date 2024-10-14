import torch
from torch.utils.tensorboard import SummaryWriter

import mnn.vision.image_size
from mnn.vision.models.vision_transformer.e2e import MyVisionTransformer
import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.loader


from mnn.vision.dataset.coco.training.utils import *
from mnn.vision.dataset.coco.training.session import train_one_epoch, val_once


class VitObjectDetectionNetwork(torch.nn.Module):

    def __init__(self, model_config: mnn_encoder_config.MyBackboneVitConfiguration):
        super().__init__()
        expected_image_width = model_config.rgb_combinator_config.d_model
        expected_image_height = (
            model_config.rgb_combinator_config.feed_forward_dimensions
        )
        self.expected_image_size = mnn.vision.image_size.ImageSize(
            width=expected_image_width, height=expected_image_height
        )
        self.encoder = MyVisionTransformer(model_config, image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


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

    batch_size = hyperparameters_config.batch_size
    embedding_size = model_config.rgb_combinator_config.d_model
    sequence_length = model_config.rgb_combinator_config.feed_forward_dimensions
    image_size = mnn.vision.image_size.ImageSize(
        width=embedding_size, height=sequence_length
    )

    hidden_dim = embedding_size
    image_RGB = torch.rand(batch_size, 3, image_size.height, image_size.width) * 255

    object_detection_model = VitObjectDetectionNetwork(model_config=model_config)

    print(f"---------- MODEL ARCHITECTURE ------------")
    print(object_detection_model)
    print(
        f"Created model with {count_parameters(object_detection_model) / (10 ** 6)} million parameters"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    object_detection_model.to(
        device=device, dtype=hyperparameters_config.floating_point_precision
    )
    validation_image = prepare_validation_image(
        validation_image_path, object_detection_model.expected_image_size
    ).to(device, dtype=hyperparameters_config.floating_point_precision)
    temp_out = object_detection_model(validation_image)
    write_image_with_mask(temp_out, validation_image, f"epoch_{0}_id_test")

    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/"
    )

    # See coco["categories"]
    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 88]
    train_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir,
        "train",
        object_detection_model.expected_image_size,
        classes=classes,
    )
    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", object_detection_model.expected_image_size, classes=classes
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hyperparameters_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hyperparameters_config.batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(
        object_detection_model.parameters(), lr=hyperparameters_config.learning_rate
    )
    loss_fn = FocalLoss()

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/experiment7_coco_my_vit_normed_predictions")
    print("- Open tensorboard with:\ntensorboard --logdir=runs")
    for epoch in range(hyperparameters_config.epochs):
        print(f"---------- EPOCH-{epoch} ------------")
        train_one_epoch(
            train_loader,
            object_detection_model,
            optimizer,
            loss_fn,
            hyperparameters_config,
            epoch,
            io_transform=None,
            prediction_transform=None,
            device=device,
            validation_image_path=validation_image_path,
            writer=writer,
            log_rate=50,
        )
        torch.save(object_detection_model.state_dict(), "exp7_object_detection.pth")
        val_once(
            val_loader,
            object_detection_model,
            loss_fn,
            hyperparameters_config,
            epoch,
            io_transform=None,
            prediction_transform=None,
            device=device,
            writer=writer,
            log_rate=50,
        )
