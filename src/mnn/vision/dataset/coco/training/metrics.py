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
