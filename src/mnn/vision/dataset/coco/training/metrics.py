import torch


def calculate_iou_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    threshold: float = 0.2,
):
    """
    Calculate the Intersection over Union (IoU) for a batch of binary masks.

    Args:
        preds (torch.Tensor): Predicted masks of shape (N, H, W) with values in [0, 1].
        targets (torch.Tensor): Ground truth masks of shape (N, H, W) with values in {0, 1}.
        smooth (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU score for each image in the batch.
    """
    # Convert predictions to binary (0 or 1)
    preds = (preds > threshold).float()

    # Calculate intersection and union for each image in the batch
    intersection = (preds * targets).sum(dim=(1, 2))
    total = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

    union = total - intersection

    # Calculate IoU for each image
    iou = (intersection + smooth) / (
        union + smooth
    )  # Adding smooth to avoid division by zero
    return iou


def calculate_iou_bbox_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) for a batch of bounding boxes.

    Args:
        preds (torch.Tensor): Predicted bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format.
        targets (torch.Tensor): Ground truth bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format.
        smooth (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU score for each bounding box in the batch.
    """
    # Calculate the intersection rectangle
    x1 = torch.max(preds[:, 0], targets[:, 0])
    y1 = torch.max(preds[:, 1], targets[:, 1])
    x2 = torch.min(preds[:, 2], targets[:, 2])
    y2 = torch.min(preds[:, 3], targets[:, 3])

    # Calculate the area of intersection rectangle
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate the area of both bounding boxes
    area_pred = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    area_target = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])

    # Calculate the union area
    union = area_pred + area_target - intersection

    # Calculate IoU for each bounding box
    iou = (intersection + smooth) / (
        union + smooth
    )  # Adding smooth to avoid division by zero
    return iou


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    threshold: float = 0.2,
):
    """
    Calculate the Dice coefficient for a batch of binary masks.

    Args:
        preds (torch.Tensor): Predicted masks of shape (N, H, W) with values in [0, 1].
        targets (torch.Tensor): Ground truth masks of shape (N, H, W) with values in {0, 1}.
        smooth (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: Dice coefficient for each image in the batch.
    """
    # Convert predictions to binary (0 or 1)
    preds = (preds > threshold).float()

    # Calculate intersection and union for each image in the batch
    intersection = (preds * targets).sum(dim=(1, 2))
    total = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

    # Calculate Dice coefficient for each image
    dice = (2 * intersection + smooth) / (
        total + smooth
    )  # Adding smooth to avoid division by zero
    return dice
