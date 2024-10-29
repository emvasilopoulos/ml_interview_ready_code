import argparse
import pathlib
from typing import List, Optional
import logging

import torch

import mnn.vision.image_size
import mnn.vision.dataset.coco.experiments.detection_ordinal as mnn_ordinal
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.model as mnn_vit_model
import mnn.vision.models.vision_transformer.ready_architectures.experiment1.config as mnn_vit_config
import mnn.vision.dataset.coco.training.utils as mnn_train_utils
import mnn.logging

LOGGER = mnn.logging.get_logger(__name__)


def load_dataset(
    dataset_dir: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
    classes: Optional[List[int]] = None,
) ->mnn_ordinal.COCOInstances2017Ordinal:
    val_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir, "val", expected_image_size
    )
    return val_dataset


def load_model(
    config_path: pathlib.Path, existing_model_path: Optional[pathlib.Path] = None
) -> mnn_vit_model.VitObjectDetectionNetwork:
    model_config, _, head_config = mnn_vit_config.load_model_config(config_path)
    model = mnn_vit_model.VitObjectDetectionNetwork(
        model_config=model_config, head_config=head_config, head_activation=torch.nn.Sigmoid()
    )
    state_dict = model.state_dict()
    if existing_model_path:
        state_dict = torch.load(existing_model_path)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=False,
        default="/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=False,
        default="/home/emvasilopoulos/projects/ml_interview_ready_code/experiments/coco/instances/experiment1/model.yaml",
    )
    parser.add_argument("--weights", type=str, required=True, default=None)
    args = parser.parse_args()

    LOGGER.info("------ LOADING ------")
    # MODEL
    model_config_path = pathlib.Path(args.model_config)
    if args.weights is not None:
        existing_model_path = pathlib.Path(args.weights)
        LOGGER.info(f"Existing model: {args.weights}")
    else:
        existing_model_path = None
    model = load_model(model_config_path, existing_model_path)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    # DATASET
    LOGGER.info("dataset...")
    dataset_dir = pathlib.Path(args.dataset_dir)
    expected_image_size = model.expected_image_size
    classes = None  # ALL CLASSES
    val_dataset = load_dataset(dataset_dir, expected_image_size)

    # Validation Image
    image, target = val_dataset[0]
    image = image.to(device=device)
    predictions = output = model(image.unsqueeze(0))
    bboxes, categories, confidence_scores = mnn_ordinal.decode_output_tensor(predictions.squeeze(0), filter_by_objectness_score=False)
    mnn_train_utils.write_image_with_output_of_experiment2(
        predictions, image, "prediction"
    )
