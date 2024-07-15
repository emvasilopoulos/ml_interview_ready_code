from typing import List
import dataclasses

import torch
import mnn.vision.dataset.word_detection.bounding_client_rect as bounding_client_rect
from mnn.vision.models.vision_transformer.encoder.vit_encoder import (
    RawVisionTransformerMultiChannelEncoder,
)


@dataclasses.dataclass
class TopLeftWidthHeightRectangle:
    top: int
    left: int
    width: int
    height: int


class VisionTransformerHead(torch.nn.Module):

    def __init__(self, previous_encoder: RawVisionTransformerMultiChannelEncoder):

        pass


class ObjectDetectionOrdinalTransformation:
    """
    A class transforming the ground truth data into the training format
    for the object detection head.
    It should be used in combination with binary-cross-entropy loss.
    It's worth trying focal loss as well.
    """

    INNER_EXPANSION = 10  # in pixels
    OUTTER_EXPANSION = 5  # in pixels
    # outter_expansion <= inner_expansion ALWAYS,
    # otherwise when two rectangles are close to each other
    # the mask will be ruined

    @staticmethod
    def expand_rectangle_outwards(
        mask: torch.Tensor, y: int, x: int, height: int, width: int
    ):
        """
        Same as 'encode_rectangle' but faster by exploiting PyTorch operations
        """
        expansion_size = ObjectDetectionOrdinalTransformation.OUTTER_EXPANSION
        for i in range(0, expansion_size):
            probability = expansion_size - i / expansion_size
            if y - i >= 0:
                mask[y - i, x : x + width] = torch.where(
                    mask[y - i, x : x + width] < probability,
                    probability,
                    mask[y - i, x : x + width],
                )
            if y + height + i < mask.shape[0]:
                mask[y + height + i, x : x + width] = torch.where(
                    mask[y + height + i, x : x + width] < probability,
                    probability,
                    mask[y + height + i, x : x + width],
                )
            if x - i >= 0:
                mask[y : y + height, x - i] = torch.where(
                    mask[y : y + height, x - i] < probability,
                    probability,
                    mask[y : y + height, x - i],
                )
            if x + width + i < mask.shape[1]:
                mask[y : y + height, x + width + i] = torch.where(
                    mask[y : y + height, x + width + i] < probability,
                    probability,
                    mask[y : y + height, x + width + i],
                )

            # TODO - Fix Corners
            for j in range(0, expansion_size):
                # Top Left Corner
                if y - i >= 0 and x - i >= 0:
                    mask[y - i + j, x - i] = torch.where(
                        mask[y - i + j, x - i] < probability,
                        probability,
                        mask[y - i + j, x - i],
                    )
                if y - i >= 0 and x - i >= 0:
                    mask[y - i, x - i + j] = torch.where(
                        mask[y - i, x - i + j] < probability,
                        probability,
                        mask[y - i, x - i + j],
                    )
                # Top Right Corner
                if y - i >= 0 and x + width + i < mask.shape[1]:
                    mask[y - i + j, x + width + i] = torch.where(
                        mask[y - i + j, x + width + i] < probability,
                        probability,
                        mask[y - i + j, x + width + i],
                    )
                if y - i >= 0 and x + width + i < mask.shape[1]:
                    mask[y - i, x + width + i - j] = torch.where(
                        mask[y - i, x + width + i - j] < probability,
                        probability,
                        mask[y - i, x + width + i - j],
                    )
                # Bottom Left Corner
                if y + height + i < mask.shape[0] and x - i >= 0:
                    mask[y + height + i - j, x - i] = torch.where(
                        mask[y + height + i - j, x - i] < probability,
                        probability,
                        mask[y + height + i - j, x - i],
                    )
                if y + height + i < mask.shape[0] and x - i >= 0:
                    mask[y + height + i, x - i + j] = torch.where(
                        mask[y + height + i, x - i + j] < probability,
                        probability,
                        mask[y + height + i, x - i + j],
                    )
                # Bottom Right Corner
                if y + height + i < mask.shape[0] and x + width + i < mask.shape[1]:
                    mask[y + height + i - j, x + width + i] = torch.where(
                        mask[y + height + i - j, x + width + i] < probability,
                        probability,
                        mask[y + height + i - j, x + width + i],
                    )
                if y + height + i < mask.shape[0] and x + width + i < mask.shape[1]:
                    mask[y + height + i, x + width + i - j] = torch.where(
                        mask[y + height + i, x + width + i - j] < probability,
                        probability,
                        mask[y + height + i, x + width + i - j],
                    )

    @staticmethod
    def expand_rectangle_inwards(
        mask: torch.Tensor, y: int, x: int, height: int, width: int
    ):
        """
        Encode the rectangle into the mask.
        """
        inner_expansion = ObjectDetectionOrdinalTransformation.INNER_EXPANSION
        for i in range(1, inner_expansion):
            probability = inner_expansion - i / inner_expansion
            if x + i < mask.shape[1]:
                mask[y + 1 : y + height, x + i] = torch.where(
                    mask[y + 1 : y + height, x + i] < probability,
                    probability,
                    mask[y + 1 : y + height, x + i],
                )
            if x + width - i >= 0:
                mask[y + 1 : y + height, x + width - i] = torch.where(
                    mask[y + 1 : y + height, x + width - i] < probability,
                    probability,
                    mask[y + 1 : y + height, x + width - i],
                )
            if y + i < mask.shape[0]:
                mask[y + i, x + 1 : x + width] = torch.where(
                    mask[y + i, x + 1 : x + width] < probability,
                    probability,
                    mask[y + i, x + 1 : x + width],
                )
            if y + height - i >= 0:
                mask[y + height - i, x + 1 : x + width] = torch.where(
                    mask[y + height - i, x + 1 : x + width] < probability,
                    probability,
                    mask[y + height - i, x + 1 : x + width],
                )

    @staticmethod
    def encode_rectangle(mask, y: int, x: int, height: int, width: int):
        ObjectDetectionOrdinalTransformation.expand_rectangle_inwards(
            mask, y, x, height, width
        )
        ObjectDetectionOrdinalTransformation.expand_rectangle_outwards(
            mask, y, x, height, width
        )

    @staticmethod
    def transform_ground_truth(
        mask_shape: torch.Size,
        original_image_shape: torch.Size,
        rectangles: List[TopLeftWidthHeightRectangle],
    ):
        mask = torch.zeros(mask_shape)  # (H, W)

        for rectangle in rectangles:
            y_normalized = rectangle.top / original_image_shape[0]
            x_normalized = rectangle.left / original_image_shape[1]
            height_normalized = rectangle.height / original_image_shape[0]
            width_normalized = rectangle.width / original_image_shape[1]

            mask_y = int(y_normalized * mask_shape[0])
            mask_x = int(x_normalized * mask_shape[1])
            mask_height = int(height_normalized * mask_shape[0])
            mask_width = int(width_normalized * mask_shape[1])

            ObjectDetectionOrdinalTransformation.encode_rectangle(
                mask, mask_y, mask_x, mask_height, mask_width
            )
        return mask


#  Autogenerated - will study later
# class OneObjectOrdinalRegressionHead(nn.Module):
#     def __init__(self, in_channels, num_classes, num_anchors, num_ordinal_bins, num_layers=4, num_filters=256):
#         super(OneObjectOrdinalRegressionHead, self).__init__()
#         self.num_classes = num_classes
#         self.num_anchors = num_anchors
#         self.num_ordinal_bins = num_ordinal_bins
#         self.num_layers = num_layers
#         self.num_filters = num_filters

#         self.conv_layers = nn.ModuleList()
#         self.conv_layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
#         for _ in range(num_layers - 1):
#             self.conv_layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
#         self.conv_layers.append(nn.Conv2d(num_filters, num_anchors * (num_classes + num_ordinal_bins), kernel_size=3, padding=1))

#     def forward(self, x):
#         for layer in self.conv_layers:
#             x = F.relu(layer(x))
#         return x

#     def loss(self, pred, target):
#         pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes + self.num_ordinal_bins)
#         target = target.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes + self.num_ordinal_bins)

#         # classification loss
#         cls_pred = pred[:, :self.num_classes]
#         cls_target = target[:, :self.num_classes]
#         cls_loss = F.cross_entropy(cls_pred, cls_target.argmax(dim=1), reduction='none')

#         # ordinal regression loss
#         ord_pred = pred[:, self.num_classes:]
#         ord_target = target[:, self.num_classes:]
#         ord_loss = F.cross_entropy(ord_pred, ord_target.argmax(dim=1), reduction='none')

#         return cls_loss, ord_loss

#     def decode(self, pred, score_thresh=0.5, nms_thresh=0.5):
#         pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes + self.num_ordinal_bins)
#         cls_pred = pred[:, :self.num_classes]
#         ord_pred = pred[:, self.num_classes:]

#         cls_scores = F.softmax(cls_pred, dim=1)

if __name__ == "__main__":
    mask_shape = torch.Size([500, 500])
    original_image_shape = torch.Size([1000, 1000])
    rectangles = [
        TopLeftWidthHeightRectangle(100, 100, 200, 200),
        TopLeftWidthHeightRectangle(250, 302, 100, 100),
        TopLeftWidthHeightRectangle(0, 0, 50, 50),
        TopLeftWidthHeightRectangle(0, 900, 100, 100),
        TopLeftWidthHeightRectangle(900, 0, 100, 100),
        TopLeftWidthHeightRectangle(900, 900, 100, 100),
    ]
    mask = ObjectDetectionOrdinalTransformation.transform_ground_truth(
        mask_shape, original_image_shape, rectangles
    )
    mask_as_numpy = mask.numpy()
    mask_as_numpy = mask_as_numpy * 255
    mask_as_numpy = mask_as_numpy.astype("uint8")
    import cv2

    cv2.imshow("mask", mask_as_numpy)
    cv2.waitKey(0)
