import numpy as np
import torch

from mnn.vision.models.heads.object_detection import *
import mnn.vision.dataset.utilities
import mnn.vision.dataset.coco.torch_dataset


from mnn.vision.dataset.coco.training.utils import *
from experiment16.train import *


def torch_mask_to_cv(mask: torch.Tensor) -> np.ndarray:
    return (mask.detach().cpu().numpy() * 255).astype("uint8")


if __name__ == "__main__":
    model_config, encoder_config, head_config = load_model_config(
        pathlib.Path("experiment16/model.yaml")
    )
    hyperparameters_config = load_hyperparameters_config(
        pathlib.Path("experiment16/hyperparameters.yaml")
    )

    dataset_dir = pathlib.Path("/home/manos/ml_interview_ready_code/data")
    image_size = mnn.vision.image_size.ImageSize(640, 480, 3)
    val_dataset = mnn.vision.dataset.coco.loader.COCODatasetInstances2017(
        dataset_dir, "val", image_size, classes=None
    )
    mask_target: torch.Tensor
    image, mask_target = val_dataset[1]
    mask_cv = torch_mask_to_cv(mask_target)
    print(mask_cv[:10, :10])

    cv2.imwrite(f"pred_mask.jpg", cv2.cvtColor(mask_cv, cv2.COLOR_GRAY2BGR))
    print(np.unique(mask_cv))
    # for x in mask_target.unique():
    #     print(x)
