import torch
from torchvision.ops import complete_box_iou

# Example bounding boxes
bboxes1 = torch.tensor(
    [[10, 20, 30, 40], [15, 25, 35, 45], [50, 60, 70, 80]]
)  # 3 boxes
bboxes2 = torch.tensor(
    [
        [10, 20, 30, 40],
        [15, 25, 35, 45],
    ]
)  # 2 boxes

# Compute pairwise IoU
iou_matrix = complete_box_iou(bboxes1, bboxes2)  # Shape: [3, 2]

# Reduce IoU matrix to handle dimension mismatch
# Example: Assign each box in bboxes1 to its best match in bboxes2
print(iou_matrix)
max_iou_per_bbox1, _ = iou_matrix.max(
    dim=1
)  # Shape: [3] - max IoU for each bbox in bboxes1

print(max_iou_per_bbox1)
# Define IoU loss: Complete IoU loss could be 1 - IoU
iou_loss = 1 - max_iou_per_bbox1.mean()  # Scalar loss
print("IoU Loss:", iou_loss.item())
