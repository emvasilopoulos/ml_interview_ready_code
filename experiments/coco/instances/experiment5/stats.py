import argparse
import pathlib
from typing import Tuple
import os

import torch
import torch.utils.tensorboard
import matplotlib.pyplot as plt

import mnn.logging
import mnn.lr_scheduler
import mnn.vision.dataset.coco.experiments.ordinal.detection_ordinal as mnn_ordinal
import mnn.vision.image_size
import mnn.vision.models.cnn.object_detection as mnn_vit_model

LOGGER = mnn.logging.get_logger(__name__)


def load_datasets(
    dataset_dir: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
    output_shape: mnn.vision.image_size.ImageSize,
) -> Tuple[
    mnn_ordinal.COCOInstances2017Ordinal,
    mnn_ordinal.COCOInstances2017Ordinal,
]:
    train_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir, "train", expected_image_size, output_shape
    )
    val_dataset = mnn_ordinal.COCOInstances2017Ordinal(
        dataset_dir, "val", expected_image_size, output_shape
    )
    return train_dataset, val_dataset


def load_model() -> mnn_vit_model.Vanilla:
    image_size = mnn.vision.image_size.ImageSize(576, 576)
    model = mnn_vit_model.Vanilla(image_size)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=False,
        default="/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco",
    )

    args = parser.parse_args()

    LOGGER.info("------ LOADING ------")
    # MODEL
    model = load_model()
    initial_epoch = model.state_dict().get("epoch", 0)
    LOGGER.info(f"Initial epoch: {initial_epoch}")
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
    train_dataset, val_dataset = load_datasets(
        dataset_dir, expected_image_size, model.output_shape
    )

    _, target = train_dataset[0]
    (
        target_xc_ordinals,
        target_yc_ordinals,
        target_w_ordinals,
        target_h_ordinals,
        target_objectness_scores,
        target_class_scores,
    ) = train_dataset.split_output_to_vectors(target)
    xcs = torch.zeros_like(target_xc_ordinals)
    ycs = torch.zeros_like(target_yc_ordinals)
    ws = torch.zeros_like(target_w_ordinals)
    hs = torch.zeros_like(target_h_ordinals)
    objectness_scores = torch.zeros_like(target_objectness_scores)
    class_scores = torch.zeros_like(target_class_scores)
    for i, (image, target) in enumerate(train_dataset):
        (
            target_xc_ordinals,
            target_yc_ordinals,
            target_w_ordinals,
            target_h_ordinals,
            target_objectness_scores,
            target_class_scores,
        ) = train_dataset.split_output_to_vectors(target)

        xcs += target_xc_ordinals
        ycs += target_yc_ordinals
        ws += target_w_ordinals
        hs += target_h_ordinals
        objectness_scores += target_objectness_scores
        class_scores += target_class_scores
        if i % 100 == 0:
            LOGGER.info(f"Processed {i} images")
    LOGGER.info(f"Processed {i} images | Done")

    xcs_total = xcs.sum(dim=0)
    ycs_total = ycs.sum(dim=0)
    ws_total = ws.sum(dim=0)
    hs_total = hs.sum(dim=0)
    class_scores_total = class_scores.sum(dim=0)

    os.makedirs("stats", exist_ok=True)
    # Plot each
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(xcs_total)
    axs[0, 0].set_title("Xc")
    axs[0, 1].plot(ycs_total)
    axs[0, 1].set_title("Yc")
    axs[1, 0].plot(ws_total)
    axs[1, 0].set_title("W")
    axs[1, 1].plot(hs_total)
    axs[1, 1].set_title("H")

    # Save figure
    plt.savefig("stats/ordinals.png")
    plt.close()

    # Plot class scores
    fig, ax = plt.subplots()
    ax.plot(class_scores_total)
    ax.set_title("Class scores")
    plt.savefig("stats/class_scores.png")
    plt.close()

    # Plot objectness scores
    fig, ax = plt.subplots()
    ax.plot(objectness_scores)
    ax.set_title("Objectness scores")
    plt.savefig("stats/objectness_scores.png")
    plt.close()
