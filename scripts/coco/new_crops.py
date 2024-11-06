import argparse
import pandas as pd
import pathlib
import random

random.seed(42)


def map_bboxes_to_cropped_image(
    row, start_x_percentage, start_y_percentage, end_x_percentage, end_y_percentage
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
    w = row["w_norm"] * image_width
    x1 = row["x1_norm"] * image_width
    h = row["h_norm"] * image_height
    y1 = row["y1_norm"] * image_height

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

    xc = new_x1 + new_w1 / 2
    yc = new_y1 + new_h1 / 2
    row["start_x.crop"] = X1
    row["start_y.crop"] = Y1
    row["end_x.crop"] = X2
    row["end_y.crop"] = Y2
    row["x1_norm.crop"] = xc / (new_image_width)
    row["y1_norm.crop"] = yc / (new_image_height)
    row["w_norm.crop"] = new_w1 / (new_image_width)
    row["h_norm.crop"] = new_h1 / (new_image_height)
    return row


class Counter:
    val = 0


def modify_group(group):
    start_x_percentage = random.uniform(0.10, 0.43)
    start_y_percentage = random.uniform(0.10, 0.43)
    end_x_percentage = random.uniform(0.57, 0.9)
    end_y_percentage = random.uniform(0.57, 0.9)
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
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)

    csv_path = pathlib.Path(args.annotations_csv_path)
    csv_dir = csv_path.parent
    csv_name = csv_path.stem

    df = pd.read_csv(csv_path)
    df2 = df.groupby(by=["image_id"]).apply(lambda group: modify_group(group))

    new_csv_path = csv_dir / f"{csv_name}_random_seed_{args.random_seed}.csv"
    df2.to_csv(new_csv_path, index=False)
