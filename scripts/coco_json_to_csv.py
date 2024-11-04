import pathlib

import pandas as pd

from mnn.vision.image_size import ImageSize
from mnn.vision.dataset.coco.torch_dataset import COCODatasetInstances2017

if __name__ == "__main__":
    dataset_dir = pathlib.Path("/home/manos/ml_interview_ready_code/data/")
    expected_image_size = ImageSize(width=640, height=480)
    splits = ["val", "train"]
    for split in splits:
        dataset = COCODatasetInstances2017(
            dataset_dir, split, expected_image_size=expected_image_size
        )

        iscrowds = []
        image_ids = []
        xcs = []
        ycs = []
        ws = []
        hs = []
        category_ids = []
        ids = []
        xc_norms = []
        yc_norms = []
        w_norms = []
        h_norms = []
        for i in range(len(dataset)):
            if i % 1000 == 0:
                print(f"Processing image {i} / {len(dataset)}")
            image, annotations = dataset.get_pair(i)
            for annotation in annotations:
                x1, y1, w, h = annotation["bbox"]
                xc = x1 + w / 2
                yc = y1 + h / 2

                x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]
                xc_norm = x1_norm + w_norm / 2
                yc_norm = y1_norm + h_norm / 2

                iscrowds.append(annotation["iscrowd"])
                image_ids.append(annotation["image_id"])
                xcs.append(xc)
                ycs.append(yc)
                ws.append(w)
                hs.append(h)
                category_ids.append(annotation["category_id"])
                ids.append(annotation["id"])
                xc_norms.append(xc_norm)
                yc_norms.append(yc_norm)
                w_norms.append(w_norm)
                h_norms.append(h_norm)

        df = pd.DataFrame(
            {
                "id": ids,
                "image_id": image_ids,
                "category_id": category_ids,
                "iscrowd": iscrowds,
                "xc": xcs,
                "yc": ycs,
                "w": ws,
                "h": hs,
                "xc_norm": xc_norms,
                "yc_norm": yc_norms,
                "w_norm": w_norms,
                "h_norm": h_norms,
            }
        )
        df.to_csv(f"{split}_annotations.csv", index=False)
