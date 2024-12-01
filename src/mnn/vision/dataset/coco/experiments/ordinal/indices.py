import torch


class VectorIndices:

    def __init__(self, vector_size: int, n_classes: int):
        self.vector_size = vector_size
        self.n_classes = n_classes

        self.bbox_vector_size = vector_size - (n_classes + 1)
        if self.bbox_vector_size % 4 != 0:
            raise ValueError(
                f"The number of elements in the vector of size {self.bbox_vector_size} must be divisible by 4 for the bounding box . Make sure you (input_image.width - 80) is divisible by 4."
            )
        self.coord_len = self.bbox_vector_size // 4

        self.objectness_idx = self.bbox_vector_size
        self.classes_start_idx = self.bbox_vector_size + 1

    def split_mask_batch(self, y: torch.Tensor):
        xc_ordinals = y[:, :, 0 : self.coord_len]
        yc_ordinals = y[:, :, self.coord_len : 2 * self.coord_len]
        w_ordinals = y[:, :, 2 * self.coord_len : 3 * self.coord_len]
        h_ordinals = y[:, :, 3 * self.coord_len : 4 * self.coord_len]
        objectness_scores = y[:, :, self.objectness_idx]
        class_scores = y[:, :, self.classes_start_idx :]
        return (
            xc_ordinals,
            yc_ordinals,
            w_ordinals,
            h_ordinals,
            objectness_scores,
            class_scores,
        )

    def split_mask(self, y: torch.Tensor):
        xc_ordinals = y[:, 0 : self.coord_len]
        yc_ordinals = y[:, self.coord_len : 2 * self.coord_len]
        w_ordinals = y[:, 2 * self.coord_len : 3 * self.coord_len]
        h_ordinals = y[:, 3 * self.coord_len : 4 * self.coord_len]
        objectness_scores = y[:, self.objectness_idx]
        class_scores = y[:, self.classes_start_idx :]
        return (
            xc_ordinals,
            yc_ordinals,
            w_ordinals,
            h_ordinals,
            objectness_scores,
            class_scores,
        )

    def split_vector(self, y: torch.Tensor):
        xc_ordinals = y[0 : self.coord_len]
        yc_ordinals = y[self.coord_len : 2 * self.coord_len]
        w_ordinals = y[2 * self.coord_len : 3 * self.coord_len]
        h_ordinals = y[3 * self.coord_len : 4 * self.coord_len]
        objectness_scores = y[self.objectness_idx]
        class_scores = y[self.classes_start_idx :]
        return (
            xc_ordinals,
            yc_ordinals,
            w_ordinals,
            h_ordinals,
            objectness_scores,
            class_scores,
        )

    def get_objectnesses(self, y: torch.Tensor):
        return y[:, self.objectness_idx]

    def get_objectnesses_batch(self, y: torch.Tensor):
        return y[:, :, self.objectness_idx]
