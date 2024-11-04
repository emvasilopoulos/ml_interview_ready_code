import pathlib
import random
import matplotlib.pyplot as plt
import pandas as pd

from mnn.vision.dataset.coco.experiments.detection_ordinal import (
    COCOInstances2017Ordinal,
)
from mnn.vision.image_size import ImageSize
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio


class Dataset(COCOInstances2017Ordinal):
    pass


def _random_percentage():
    side = random.randint(0, 1)
    if side == 0:
        return random.random() * 0.35
    else:
        return random.random() * 0.35 + 0.65


if __name__ == "__main__":
    dataset_dir = pathlib.Path("/home/manos/ml_interview_ready_code/data/")
    expected_image_size = ImageSize(width=640, height=480)
    dataset = Dataset(dataset_dir, "train", expected_image_size=expected_image_size)

    xcs = []
    ycs = []
    ws = []
    hs = []
    pad_dim1_counter = 0
    pad_dim2_counter = 0

    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(
                f"Processing image {i} / {len(dataset)} | pad_dim1: {pad_dim1_counter} | pad_dim2: {pad_dim2_counter}"
            )
        image, annotations = dataset.get_pair(i)
        # Calculate new dimensions & resize only
        current_image_size = ImageSize(width=image.shape[2], height=image.shape[1])
        fixed_ratio_components = mnn_resize_fixed_ratio.calculate_new_tensor_dimensions(
            current_image_size, expected_image_size
        )
        if fixed_ratio_components.pad_dimension == 1:
            pad_dim1_counter += 1
        else:
            pad_dim2_counter += 1

        for annotation in annotations:
            x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]
            xc = x1_norm + w_norm / 2
            yc = y1_norm + h_norm / 2

            area = w_norm * h_norm
            # Skip very small bboxes. Bad annotations
            if area < 0.0004:
                continue
            # Skip very close to image borders bboxes. Bad annotations
            if (
                x1_norm > 0.99
                or y1_norm > 0.99
                or (x1_norm + w_norm) <= 0.01
                or (y1_norm + h_norm) <= 0.01
            ):
                continue

            x1 = x1_norm * fixed_ratio_components.resize_width
            y1 = y1_norm * fixed_ratio_components.resize_height
            w = w_norm * fixed_ratio_components.resize_width
            h = h_norm * fixed_ratio_components.resize_height
            padding_percent = _random_percentage()
            x1, y1, x2, y2 = dataset.map_bbox_to_padded_image(
                x1, y1, w, h, fixed_ratio_components, padding_percent
            )

            w = x2 - x1
            h = y2 - y1
            xc = x1 + w / 2
            yc = y1 + h / 2

            xcs.append(xc)
            ycs.append(yc)
            ws.append(w_norm)
            hs.append(h_norm)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[0, 0].hist(xcs, bins=100)
    axs[0, 0].set_title("X center")
    axs[0, 1].hist(ycs, bins=100)
    axs[0, 1].set_title("Y center")
    axs[1, 0].hist(ws, bins=100)
    axs[1, 0].set_title("Width")
    axs[1, 1].hist(hs, bins=100)
    axs[1, 1].set_title("Height")

    # store the plot
    plt.savefig("bbox_histograms.png")
