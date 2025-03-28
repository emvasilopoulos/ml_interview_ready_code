import pathlib
import os
import random

import cv2
import torch
import torch.utils.tensorboard

import mnn.vision.image_size
import mnn.vision.dataset.object_detection.fading_bboxes_in_mask


def prepare_validation_image(
    validation_image_path: pathlib.Path,
    expected_image_size: mnn.vision.image_size.ImageSize,
):

    preprocessor = (
        mnn.vision.dataset.object_detection.fading_bboxes_in_mask.FadingBboxMasks
    )
    img = cv2.imread(validation_image_path.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padding_percent = random.random()
    val_img_tensor = preprocessor.cv2_image_to_tensor(img)
    val_img_tensor = preprocessor.preprocess_image(
        val_img_tensor,
        expected_image_size,
        padding_percent=padding_percent,
    )
    return val_img_tensor.unsqueeze(0)


def write_image_with_output_of_experiment1(
    temp_out: torch.Tensor,
    validation_image: torch.Tensor,
    sub_dir: str = "any",
):
    validation_img = validation_image.squeeze(0).detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    temp_out = temp_out.squeeze(0).detach().cpu()

    image = (validation_img.numpy() * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # reverse mask
    raw_mask = (temp_out.numpy() * 255).astype("uint8")
    mask = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.addWeighted(image, 1, mask, 0.7, 0)
    os.makedirs(f"assessment_images/{sub_dir}", exist_ok=True)
    cv2.imwrite(f"assessment_images/{sub_dir}/raw_mask.jpg", raw_mask)
    cv2.imwrite(f"assessment_images/{sub_dir}/masked_image.jpg", masked_image)


def write_image_with_output_of_experiment2(
    temp_out: torch.Tensor,
    validation_image: torch.Tensor,
    sub_dir: str = "any",
):
    bboxes, categories, confidence_scores = decode_output_tensor(
        temp_out.squeeze(0), filter_by_objectness_score=False
    )

    validation_img = validation_image.squeeze(0).detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    image = (validation_img.numpy() * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for bbox, category, confidence in zip(bboxes, categories, confidence_scores):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{category.item()} - {confidence:.3f}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # reverse mask
    os.makedirs(f"assessment_images/{sub_dir}", exist_ok=True)
    cv2.imwrite(f"assessment_images/{sub_dir}/bboxed_image.jpg", image)
