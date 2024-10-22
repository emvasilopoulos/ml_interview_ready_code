from typing import List
import torch

import mnn.vision.image_size

import mnn.vision.dataset.utilities
import mnn.vision.models.heads.object_detection
import mnn.vision.dataset.coco.torch_dataset


from mnn.vision.dataset.coco.training.utils import *
from experiment14.train import *

if __name__ == "__main__":

    model_config, encoder_config, head_config = load_model_config(
        pathlib.Path("experiment14/model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("experiment14/hyperparameters.yaml")
    )

    model = VitObjectDetectionNetwork(model_config, head_config)
    model.load_state_dict(
        torch.load("experiment14/trained_models/exp14_object_detection.pth")
    )

    image_size = model.expected_image_size
    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
    )
    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", image_size, classes=None
    )
    image, mask = val_dataset[0]

    mask_pred = model(image.unsqueeze(0))

    bce_loss = torch.nn.BCELoss()

    loss = bce_loss(mask_pred, mask.unsqueeze(0))
    print(loss)

    import numpy as np

    mask_pred = mask_pred.squeeze(0)
    mask_pred = (mask_pred.detach().cpu().numpy() * 255).astype(np.uint8)
    cv2.imshow("mask_pred", mask_pred)
    cv2.waitKey(0)
