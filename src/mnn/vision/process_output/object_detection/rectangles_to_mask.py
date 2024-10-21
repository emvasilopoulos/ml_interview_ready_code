from typing import List
import dataclasses

import torch


@dataclasses.dataclass
class TopLeftWidthHeightRectangle:
    top: int
    left: int
    width: int
    height: int


class ObjectDetectionOrdinalTransformation:
    """
    A class transforming the ground truth data into the training format
    for the object detection head.
    It should be used in combination with binary-cross-entropy loss.
    It's worth trying focal loss as well.
    """

    INNER_EXPANSION = 5  # in pixels
    OUTTER_EXPANSION = 0  # in pixels
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
        for i in range(1, expansion_size):
            probability = round((1 - i / expansion_size) * 100) / 100
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

            # Fix Corners
            for j in range(0, expansion_size):
                # Top Left Corner
                y_ij = y - i + j
                if y_ij >= 0 and y_ij < mask.shape[0] and x - i >= 0:
                    mask[y_ij, x - i] = torch.where(
                        mask[y_ij, x - i] < probability,
                        probability,
                        mask[y_ij, x - i],
                    )

                x_ij = x - i + j
                if y - i >= 0 and x_ij >= 0 and x_ij < mask.shape[1]:
                    mask[y - i, x_ij] = torch.where(
                        mask[y - i, x_ij] < probability,
                        probability,
                        mask[y - i, x_ij],
                    )
                # Top Right Corner
                y_ij = y - i + j
                if y_ij >= 0 and y_ij < mask.shape[0] and x + width + i < mask.shape[1]:
                    mask[y_ij, x + width + i] = torch.where(
                        mask[y_ij, x + width + i] < probability,
                        probability,
                        mask[y_ij, x + width + i],
                    )

                xw_ij = x + width + i - j
                if y - i >= 0 and xw_ij >= 0 and xw_ij < mask.shape[1]:
                    mask[y - i, xw_ij] = torch.where(
                        mask[y - i, xw_ij] < probability,
                        probability,
                        mask[y - i, xw_ij],
                    )

                # Bottom Left Corner
                yh_ij = y + height + i - j
                if yh_ij >= 0 and yh_ij < mask.shape[0] and x - i >= 0:
                    mask[yh_ij, x - i] = torch.where(
                        mask[yh_ij, x - i] < probability,
                        probability,
                        mask[yh_ij, x - i],
                    )

                yh_i = y + height + i
                x_ij = x - i + j
                if yh_i < mask.shape[0] and x_ij >= 0 and x_ij < mask.shape[1]:
                    mask[yh_i, x_ij] = torch.where(
                        mask[yh_i, x_ij] < probability,
                        probability,
                        mask[yh_i, x_ij],
                    )

                # Bottom Right Corner
                yh_ij = y + height + i - j
                xw_i = x + width + i
                if yh_i < mask.shape[0] and xw_i < mask.shape[1]:
                    mask[yh_ij, xw_i] = torch.where(
                        mask[yh_ij, xw_i] < probability,
                        probability,
                        mask[yh_ij, xw_i],
                    )

                yh_i = y + height + i
                xw_ij = x + width + i - j
                if yh_i < mask.shape[0] and xw_ij >= 0 and xw_ij < mask.shape[1]:
                    mask[yh_i, xw_ij] = torch.where(
                        mask[yh_i, xw_ij] < probability,
                        probability,
                        mask[yh_i, xw_ij],
                    )
        return mask

    @staticmethod
    def expand_rectangle_inwards(
        mask: torch.Tensor, y: int, x: int, height: int, width: int
    ):
        """
        Encode the rectangle into the mask.
        """
        inner_expansion = ObjectDetectionOrdinalTransformation.INNER_EXPANSION
        mask_h, mask_w = mask.shape
        for i in range(1, inner_expansion):
            probability = round((1 - i / inner_expansion) * 100) / 100
            # VERTICAL LINES
            if x + i < mask_w:
                y_ = y + height if (y + height) < mask_h else mask_h
                mask[y:y_, x + i] = torch.where(
                    mask[y:y_, x + i] < probability,
                    probability,
                    mask[y:y_, x + i],
                )
            if x + width - i >= 0 and x + width - i < mask_w:
                y_ = y + height if (y + height) < mask_h else mask_h
                mask[y:y_, x + width - i] = torch.where(
                    mask[y:y_, x + width - i] < probability,
                    probability,
                    mask[y:y_, x + width - i],
                )

            # HORIZONTAL LINES
            if y + i < mask_h:
                x_ = x + width if (x + width) < mask_w else mask_w
                mask[y + i, x:x_] = torch.where(
                    mask[y + i, x:x_] < probability,
                    probability,
                    mask[y + i, x:x_],
                )
            if y + height - i >= 0 and y + height - i < mask_h:
                x_ = x + width if (x + width) < mask_w else mask_w
                mask[y + height - i, x:x_] = torch.where(
                    mask[y + height - i, x:x_] < probability,
                    probability,
                    mask[y + height - i, x:x_],
                )
        return mask

    @staticmethod
    def fill_rectangle(mask: torch.Tensor, y: int, x: int, height: int, width: int):
        mask[y : y + height, x : x + width] = 1
        return mask

    @staticmethod
    def draw_rectangle(mask: torch.Tensor, y: int, x: int, height: int, width: int):
        mask[y : y + height, x] = 1
        mask[y : y + height, x + width - 1] = 1
        mask[y, x : x + width] = 1
        mask[y + height - 1, x : x + width] = 1
        return mask

    @staticmethod
    def encode_rectangle(mask, y: int, x: int, height: int, width: int):
        mask = ObjectDetectionOrdinalTransformation.draw_rectangle(
            mask, y, x, height, width
        )
        mask = ObjectDetectionOrdinalTransformation.expand_rectangle_inwards(
            mask, y, x, height, width
        )
        mask = ObjectDetectionOrdinalTransformation.expand_rectangle_outwards(
            mask, y, x, height, width
        )
        return mask

    @staticmethod
    def transform(
        mask_shape: torch.Size,
        original_image_shape: torch.Size,
        rectangles: List[TopLeftWidthHeightRectangle],
    ) -> torch.Tensor:
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

            mask = ObjectDetectionOrdinalTransformation.encode_rectangle(
                mask, mask_y, mask_x, mask_height, mask_width
            )
        return mask

    @staticmethod
    def transform_from_normalized_rectangles(
        mask_shape: torch.Size,
        rectangles: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.zeros(mask_shape)  # (H, W)

        for i in range(rectangles.shape[0]):
            rectangle = rectangles[i]
            x_normalized = rectangle[0]
            y_normalized = rectangle[1]
            width_normalized = rectangle[2]
            height_normalized = rectangle[3]

            mask_y = int(y_normalized * mask_shape[0])
            mask_x = int(x_normalized * mask_shape[1])
            mask_height = int(height_normalized * mask_shape[0])
            mask_width = int(width_normalized * mask_shape[1])

            mask = ObjectDetectionOrdinalTransformation.encode_rectangle(
                mask, mask_y, mask_x, mask_height, mask_width
            )
        return mask

    @staticmethod
    def transform_ground_truth_from_categories(
        mask_shape: torch.Size,
        rectangles: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Not implemented yet")


class ObjectDetectionOrdinalTransformation_LEGACY:
    """
    This version creates a small mask for each rectangle.
    and then attaches it to the final mask.

    """

    INNER_EXPANSION = 5  # in pixels
    OUTTER_EXPANSION = 0  # in pixels

    @staticmethod
    def fill_rectangle(mask: torch.Tensor, y: int, x: int, height: int, width: int):
        mask[y : y + height, x : x + width] = 1
        return mask

    @staticmethod
    def encode_rectangle(mask: torch.Tensor, y: int, x: int, height: int, width: int):
        # Only inward expansion
        little_mask = torch.zeros((height, width))

        max_expansion = ObjectDetectionOrdinalTransformation.INNER_EXPANSION
        if height <= 2 * max_expansion or width <= 2 * max_expansion:
            # For extra small bboxes
            max_expansion = min(height, width) // 2

        for i in range(max_expansion):
            t = i
            b = height - i
            l = i
            r = width - i

            probability = round((1 - i / max_expansion) * 100) / 100

            little_mask[t:b, l] = probability  # Draw left line
            little_mask[t:b, r - 1] = probability  # Draw right line
            little_mask[t, l:r] = probability  # Draw top line
            little_mask[b - 1, l:r] = probability  # Draw bottom line
        mask[y : y + height, x : x + width] = little_mask
        return mask

    @staticmethod
    def transform_ground_truth(
        mask_shape: torch.Size,
        original_image_shape: torch.Size,
        rectangles: List[TopLeftWidthHeightRectangle],
    ) -> torch.Tensor:
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

            mask = ObjectDetectionOrdinalTransformation.encode_rectangle(
                mask, mask_y, mask_x, mask_height, mask_width
            )

        return mask

    @staticmethod
    def transform_ground_truth_from_normalized_rectangles(
        mask_shape: torch.Size,
        rectangles: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.zeros(mask_shape)  # (H, W)

        for i in range(rectangles.shape[0]):
            rectangle = rectangles[i]
            x_normalized = rectangle[0]
            y_normalized = rectangle[1]
            width_normalized = rectangle[2]
            height_normalized = rectangle[3]

            mask_y = int(y_normalized * mask_shape[0])
            mask_x = int(x_normalized * mask_shape[1])
            mask_height = int(height_normalized * mask_shape[0])
            mask_width = int(width_normalized * mask_shape[1])

            mask = ObjectDetectionOrdinalTransformation.encode_rectangle(
                mask, mask_y, mask_x, mask_height, mask_width
            )
        return mask


if __name__ == "__main__":

    def visualize_example():
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
        mask = ObjectDetectionOrdinalTransformation.transform(
            mask_shape, original_image_shape, rectangles
        )
        mask_as_numpy = mask.numpy()
        mask_as_numpy = mask_as_numpy * 255
        mask_as_numpy = mask_as_numpy.astype("uint8")
        import cv2

        cv2.imshow("mask", mask_as_numpy)
        cv2.waitKey(0)

    visualize_example()
