import torch


class FocalLoss(torch.nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 1.5, reduction: str = "mean"
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        # self.bce_loss = torch.nn.BCELoss()
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = inputs.clone()
        ce_loss = torch.nn.functional.binary_cross_entropy(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'self.reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


import torchvision


class BboxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ciou_fn = torchvision.ops.complete_box_iou_loss

    def forward(
        self,
        prediction_bboxes: torch.Tensor,
        prediction_objectnesses: torch.Tensor,
        targets_bboxes: torch.Tensor,
    ):

        if targets_bboxes.shape[0] > 0 and prediction_bboxes.shape[0] == 0:
            return 1.0
        elif targets_bboxes.shape[0] == 0 and prediction_bboxes.shape[0] > 0:
            return 1.0
        elif targets_bboxes.shape[0] == 0 and prediction_bboxes.shape[0] == 0:
            return 0
        else:
            if len(prediction_objectnesses.shape) > 1:
                raise ValueError(
                    f"prediction_objectnesses should be 1D tensor. Your shape is: {prediction_objectnesses.shape}"
                )
            #
            iou_matrix = torchvision.ops.complete_box_iou(
                prediction_bboxes, targets_bboxes
            )  # (n_pred, n_gt)

            """
            YOLOv3 paper states:
            "YOLOv3 predicts an objectness score for each bounding box using logistic regression.
            This should be 1 if the bounding box prior overlaps a ground truth object by more than
            any other bounding box prior."
            """
            iou_matrix = prediction_objectnesses.unsqueeze(1) * iou_matrix
            max_iou_per_bbox1, _ = iou_matrix.max(
                dim=1
            )  # max IoU for each bbox in predictions
            return 1 - max_iou_per_bbox1.mean()
