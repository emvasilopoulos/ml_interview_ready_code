import pathlib
import cv2
import pandas as pd


def image_name_from_coco_image_id(image_id: int) -> str:
    return f"{image_id:012}.jpg"


if __name__ == "__main__":
    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
    )
    annotations_path = dataset_dir / "annotations/instances_train2017_rand_scheme_4.csv"
    df = pd.read_csv(annotations_path.as_posix())
    hoi = [
        "start_x.crop",
        "start_y.crop",
        "end_x.crop",
        "end_y.crop",
        "x1_norm.crop",
        "y1_norm.crop",
        "w_norm.crop",
        "h_norm.crop",
    ]
    images_dir = dataset_dir / "train2017"
    for group in df.groupby("image_id"):
        image_id = group[0]
        image_name = image_name_from_coco_image_id(image_id)
        image_path = images_dir / image_name
        img = cv2.imread(image_path.as_posix())
        start_x = int(group[1].iloc[0]["start_x.crop"])
        start_y = int(group[1].iloc[0]["start_y.crop"])
        end_x = int(group[1].iloc[0]["end_x.crop"])
        end_y = int(group[1].iloc[0]["end_y.crop"])
        img_cropped = img[start_y:end_y, start_x:end_x]
        for row in group[1].iterrows():
            x1 = row[1]["x1_norm.crop"]
            y1 = row[1]["y1_norm.crop"]
            w = row[1]["w_norm.crop"]
            h = row[1]["h_norm.crop"]
            if any(k < 0 for k in [x1, y1, w, h]):
                continue
            x = int(x1 * img_cropped.shape[1])
            y = int(y1 * img_cropped.shape[0])
            width = int(w * img_cropped.shape[1])
            height = int(h * img_cropped.shape[0])
            # x -= width // 2
            # y -= height // 2
            cv2.rectangle(img_cropped, (x, y), (x + width, y + height), (0, 255, 0), 2)
        # cv2.imwrite("image.jpg", img_cropped)
        # k = input()
        # if k == "q":
        #     break
        cv2.imshow("image", img_cropped)
        if cv2.waitKey(0) == ord("q"):
            break
