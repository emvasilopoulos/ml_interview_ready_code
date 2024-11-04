import random

import matplotlib.pyplot as plt
import pandas as pd

from mnn.vision.image_size import ImageSize
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio


def _random_percentage(base=0.1):
    side = random.randint(0, 1)
    if side == 0:
        return random.random() * base
    else:
        return random.random() * base + (1 - base)


def calculate_image_width(df):
    norms_xcs = df["xc_norm"]
    xcs = df["xc"]
    widths = xcs / norms_xcs
    # Round to nearest integer
    return widths.round(decimals=0).astype(int)


def calculate_image_height(df):
    norms_ycs = df["yc_norm"]
    ycs = df["yc"]
    heights = ycs / norms_ycs
    return heights.round(decimals=0).astype(int)


import argparse
import os

HEADER = [
    "id",
    "image_id",
    "category_id",
    "iscrowd",
    "xc",
    "yc",
    "w",
    "h",
    "xc_norm",
    "yc_norm",
    "w_norm",
    "h_norm",
]


def _get_csv_name(image_size: ImageSize):
    return f"coco_train_resizeW{image_size.width}xH{image_size.height}.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--random-base", type=float, required=True)
    args = parser.parse_args()

    expected_image_size = ImageSize(width=args.width, height=args.height)
    random_base = args.random_base

    df = pd.read_csv("train_annotations.csv")

    #
    if os.path.exists(_get_csv_name(expected_image_size)):
        df = pd.read_csv(_get_csv_name(expected_image_size))
    else:
        df["image_width"] = calculate_image_width(df)
        df["image_height"] = calculate_image_height(df)

        # Calculate new dimensions
        df[
            [
                "resize_height",
                "resize_width",
                "pad_dimension",
                "expected_dimension_size",
            ]
        ] = df.apply(
            lambda x: mnn_resize_fixed_ratio.calculate_new_tensor_dimensions(
                ImageSize(width=x["image_width"], height=x["image_height"]),
                expected_image_size,
            ).as_tuple(),
            axis=1,
            result_type="expand",
        )

        df.to_csv(
            _get_csv_name(expected_image_size),
            index=False,
        )

    # calculate new bbox coordinates after padding & resizing
    def new_xc(row, percentage):
        if row["pad_dimension"] == 2:
            pad_amount = row["expected_dimension_size"] - row["resize_width"]
            return (
                row["xc_norm"] * row["resize_width"] + pad_amount * percentage
            ) / row["expected_dimension_size"]
        return row["xc_norm"]

    def new_yc(row, percentage):
        if row["pad_dimension"] == 1:
            pad_amount = row["expected_dimension_size"] - row["resize_height"]
            return (
                row["yc_norm"] * row["resize_height"] + pad_amount * percentage
            ) / row["expected_dimension_size"]
        return row["yc_norm"]

    df["xc_norm_new"] = df.apply(
        lambda row: new_xc(row, _random_percentage(random_base)), axis=1
    )
    df["yc_norm_new"] = df.apply(
        lambda row: new_yc(row, _random_percentage(random_base)), axis=1
    )

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[0, 0].hist(df["xc_norm_new"], bins=100)
    axs[0, 0].set_title("xc_norm_new")
    axs[0, 1].hist(df["yc_norm_new"], bins=100)
    axs[0, 1].set_title("yc_norm_new")
    axs[1, 0].hist(df["w_norm"], bins=100)
    axs[1, 0].set_title("w_norm")
    axs[1, 1].hist(df["h_norm"], bins=100)
    axs[1, 1].set_title("h_norm")
    plt.savefig(f"histogram_resizeW{args.width}xH{args.height}.png")
