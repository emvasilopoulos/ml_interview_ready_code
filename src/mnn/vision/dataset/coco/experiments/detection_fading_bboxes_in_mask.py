from typing import Any, Dict

import torch

from mnn.vision.dataset.coco.torch_dataset import COCODatasetInstances2017
import mnn.vision.process_input.dimensions.pad as mnn_pad
import mnn.vision.process_input.dimensions.resize_fixed_ratio as mnn_resize_fixed_ratio

import mnn.vision.process_output.object_detection.rectangles_to_mask as mnn_rectangles_to_mask


class COCOInstances2017FBM(COCODatasetInstances2017):

    bbox_to_mask_transformation = (
        mnn_rectangles_to_mask.ObjectDetectionOrdinalTransformation(4, 0, True)
    )

    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
    ) -> torch.Tensor:
        bboxes = []
        categories = []
        # Add whole image as a bounding box
        bboxes.append([0, 0, 1, 1])
        for annotation in annotations:
            category = int(annotation["category_id"])
            if (
                self.desired_classes is not None
                and category not in self.desired_classes
            ):
                continue
            x1, y1, w, h = annotation["normalized_bbox"]
            bboxes.append([x1, y1, w, h])
            categories.append(annotation["category_id"])

        # Convert bounding boxes to mask
        bboxes_as_mask = (
            self.bbox_to_mask_transformation.transform_from_normalized_rectangles(
                torch.Size(
                    (
                        fixed_ratio_components.resize_height,
                        fixed_ratio_components.resize_width,
                    )
                ),
                torch.Tensor(bboxes).float(),
            )
        )
        bboxes_as_mask = mnn_pad.pad_image(
            bboxes_as_mask.unsqueeze(0),
            pad_dimension=fixed_ratio_components.pad_dimension,
            expected_dimension_size=fixed_ratio_components.expected_dimension_size,
            padding_percent=padding_percent,
        ).squeeze(0)

        # TODO - support categories somehow
        return bboxes_as_mask


if __name__ == "__main__":
    import pathlib
    import mnn.vision.image_size
    import mnn.vision.dataset.coco.training.utils as mnn_train_utils

    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
    )
    expected_image_size = mnn.vision.image_size.ImageSize(width=512, height=512)

    val_dataset = COCOInstances2017FBM(
        dataset_dir, "val", expected_image_size, classes=None
    )
    image_batch, target0 = val_dataset[0]
    mnn_train_utils.write_image_with_mask(
        target0.unsqueeze(0),
        image_batch.unsqueeze(0),
        "train_image_ground_truth",
    )
