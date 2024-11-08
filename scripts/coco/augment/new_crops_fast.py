import argparse
import pandas as pd
import pathlib
import random
import numba

SCHEMES = [
    (0, 0.35, 0.65, 1),
    (0.1, 0.43, 0.57, 0.9),
    (0.31, 0.44, 0.85, 0.98),
    (0.26, 0.39, 0.75, 0.88),
]


def map_bboxes_to_cropped_image(
    row: pd.DataFrame,
    start_x_percentage: float,
    start_y_percentage: float,
    end_x_percentage: float,
    end_y_percentage: float,
):
    #
    image_width = row["image_width"]
    image_height = row["image_height"]

    # BBOX for cropped image
    X1 = int(image_width * start_x_percentage)
    Y1 = int(image_height * start_y_percentage)
    X2 = int(image_width * end_x_percentage)
    Y2 = int(image_height * end_y_percentage)

    # BBOX for object
    x1 = row["x1"]
    y1 = row["y1"]
    w = row["w"]
    h = row["h"]

    # new x1, y1
    new_image_width = X2 - X1
    new_image_height = Y2 - Y1
    if x1 < X1:
        new_x1 = 0
        new_w1 = w - (X1 - x1)
    elif X1 <= x1 < X2:
        new_x1 = x1 - X1
        new_w1 = w
    else:
        new_x1 = -1  # out of bounds
        new_w1 = -1

    if y1 < Y1:
        new_y1 = 0
        new_h1 = h - (Y1 - y1)
    elif Y1 <= y1 < Y2:
        new_y1 = y1 - Y1
        new_h1 = h
    else:
        new_y1 = -1
        new_h1 = -1

    # new w, h
    if new_x1 + new_w1 > X2:
        new_w1 = new_image_width - new_x1
    if new_y1 + new_h1 > Y2:
        new_h1 = new_image_height - new_y1

    row["start_x.crop"] = X1
    row["start_y.crop"] = Y1
    row["end_x.crop"] = X2
    row["end_y.crop"] = Y2
    row["x1_norm.crop"] = new_x1 / (new_image_width)
    row["y1_norm.crop"] = new_y1 / (new_image_height)
    row["w_norm.crop"] = new_w1 / (new_image_width)
    row["h_norm.crop"] = new_h1 / (new_image_height)
    return row


class Counter:
    val = 0


import pandas.api.typing


def modify_group(group: pandas.api.typing.DataFrameGroupBy, scheme: int):
    low_x, high_x, low_y, high_y = SCHEMES[scheme - 1]
    start_x_percentage = random.uniform(low_x, high_x)
    start_y_percentage = random.uniform(low_x, high_x)
    end_x_percentage = random.uniform(low_y, high_y)
    end_y_percentage = random.uniform(low_y, high_y)
    group[
        [
            "start_x.crop",
            "start_y.crop",
            "end_x.crop",
            "end_y.crop",
            "x1_norm.crop",
            "y1_norm.crop",
            "w_norm.crop",
            "h_norm.crop",
        ]
    ] = group.apply(
        lambda x: map_bboxes_to_cropped_image(
            x,
            start_x_percentage,
            start_y_percentage,
            end_x_percentage,
            end_y_percentage,
        )[
            [
                "start_x.crop",
                "start_y.crop",
                "end_x.crop",
                "end_y.crop",
                "x1_norm.crop",
                "y1_norm.crop",
                "w_norm.crop",
                "h_norm.crop",
            ]
        ],
        axis=1,
        result_type="expand",
    )
    Counter.val += 1
    if Counter.val % 1000 == 0:
        print(f"samples processed: {Counter.val}")
    return group


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-csv-path", type=str)
    parser.add_argument("--scheme", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)

    csv_path = pathlib.Path(args.annotations_csv_path)
    csv_dir = csv_path.parent
    csv_name = csv_path.stem
    df = pd.read_csv(csv_path)

    df2 = df.groupby(by=["image_id"]).apply(
        lambda group: modify_group(group, args.scheme)
    )

    new_csv_path = csv_dir / f"{csv_name}_rand_scheme_{args.scheme}.csv"
    df2.to_csv(new_csv_path, index=False)
