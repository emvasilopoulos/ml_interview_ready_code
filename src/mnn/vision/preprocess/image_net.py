import pathlib

import torchvision

import mnn.vision.preprocess.base


class ImageNetPreprocessor(mnn.vision.preprocess.base.BasePreprocessor):

    def __init__(self):
        super().__init__(
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )


if __name__ == "__main__":
    image_path = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco/val2017/000000000139.jpg"
    )
    if not image_path.exists():
        raise ValueError(f"Image path {image_path} does not exist.")
    preprocessor = ImageNetPreprocessor()
    image = preprocessor.read_image(image_path)
