import math
import torch


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(
            b2_x1
        )  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        iou -= (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return torch.nan_to_num(iou, nan=0)  # IoU


def bbox_overlaps_ciou(bboxes1: torch.Tensor, bboxes2: torch.Tensor):
    """
    expecting bboxes as TLBR
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    """

    # 1 - create array of shape (N, M) where N is number of bboxes1 and M is number of bboxes2
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    # 2 - calculate the area of each bbox
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    # 3 - calculate the center of each bbox
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    # 4 - calculate the inner and outer boxes
    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    # 5 - calculate the intersection area
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    v = (4 / (math.pi**2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v)
    cious = iou - (u + alpha * v)
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return cious


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
    print(preds.shape, targets.shape)
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
