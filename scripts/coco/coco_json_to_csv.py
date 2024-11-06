import pathlib

import cv2
import numpy as np
import pandas as pd

from mnn.vision.image_size import ImageSize
from mnn.vision.dataset.coco.torch_dataset import COCODatasetInstances2017


def image_name_from_coco_image_id(image_id: int) -> str:
    return f"{image_id:012}.jpg"


if __name__ == "__main__":
    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
    )
    expected_image_size = ImageSize(width=640, height=480)
    splits = ["val", "train"]
    for split in splits:
        dataset = COCODatasetInstances2017(
            dataset_dir, split, expected_image_size=expected_image_size
        )

        iscrowds = []
        image_ids = []
        x1s = []
        y1s = []
        ws = []
        hs = []
        category_ids = []
        ids = []
        x1_norms = []
        y1_norms = []
        w_norms = []
        h_norms = []
        images_widths = []
        images_heights = []
        for i in range(len(dataset)):
            if i % 1000 == 0:
                print(f"Processing image {i} / {len(dataset)}")
            image, annotations = dataset.get_pair(i)
            for annotation in annotations:
                x1, y1, w, h = annotation["bbox"]

                x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]

                iscrowds.append(annotation["iscrowd"])
                image_ids.append(annotation["image_id"])
                x1s.append(x1)
                y1s.append(y1)
                ws.append(w)
                hs.append(h)
                category_ids.append(annotation["category_id"])
                ids.append(annotation["id"])
                x1_norms.append(x1_norm)
                y1_norms.append(y1_norm)
                w_norms.append(w_norm)
                h_norms.append(h_norm)
                images_widths.append(image.shape[2])
                images_heights.append(image.shape[1])

        df = pd.DataFrame(
            {
                "id": ids,
                "image_id": image_ids,
                "category_id": category_ids,
                "iscrowd": iscrowds,
                "x1": x1s,
                "y1": y1s,
                "w": ws,
                "h": hs,
                "x1_norm": x1_norms,
                "y1_norm": y1_norms,
                "w_norm": w_norms,
                "h_norm": h_norms,
                "image_width": images_widths,
                "image_height": images_heights,
            }
        )
        df.to_csv(f"{split}_annotations.csv", index=False)
