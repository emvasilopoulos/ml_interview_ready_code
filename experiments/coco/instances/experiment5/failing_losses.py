class NotFailedButKeepingItExperimentalLoss(torch.nn.Module):
    counter = 0

    def __init__(self, dataset: mnn_ordinal.COCOInstances2017Ordinal3):
        self.dataset = dataset
        super().__init__()
        self.bbox_loss = FocalLoss()
        self.class_loss = FocalLoss()
        self.objectness_loss = torch.nn.BCELoss()
        self.losses = {
            "xc": 0,
            "yc": 0,
            "w": 0,
            "h": 0,
            "class": 0,
            "objectness": 0,
        }
        self.total_loss = 0

    def _total_loss(self):
        return sum(self.losses.values())

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.losses = {
            "bbox": 0,
            "class": 0,
            "objectness": 0,
        }
        for pred_sample, target_sample in zip(predictions, targets):
            (
                pred_bbox_ordinals,
                pred_class_scores,
                pred_objectnesses,
            ) = self.dataset.split_output_to_vectors2(pred_sample)

            (
                target_bbox_ordinals,
                target_class_scores,
                target_objectnesses,
            ) = self.dataset.split_output_to_vectors2(target_sample)

            self.objectness_loss_value = self.objectness_loss(
                pred_objectnesses, target_objectnesses
            )
            self.losses["objectness"] += self.objectness_loss_value

            if target_objectnesses.sum() == 0:
                continue

            vectors_positions = target_objectnesses == 1
            self.bbox_loss_value = self.bbox_loss(
                pred_bbox_ordinals[vectors_positions],
                target_bbox_ordinals[vectors_positions],
            )
            self.losses["bbox"] += self.bbox_loss_value

            self.class_loss_value = self.class_loss(
                pred_class_scores[vectors_positions],
                target_class_scores[vectors_positions],
            )
            self.losses["class"] += self.class_loss_value

        self.total_loss = self._total_loss()

        if self.counter % 50 == 0:
            pred_bbox_ordinals, pred_class_scores, pred_objectnesses = (
                self.dataset.split_output_to_vectors2(predictions[0])
            )
            target_bbox_ordinals, target_class_scores, target_objectnesses = (
                self.dataset.split_output_to_vectors2(targets[0])
            )
            if pred_objectnesses.max() < 0.05:
                pred_argmax = pred_class_scores.argmax(0)
                target_argmax = target_class_scores.argmax(0)
                print(
                    "Predictions-objectness max objectness: ",
                    pred_objectnesses[pred_argmax],
                )
                print(
                    "Targets-objectness max objectness: ",
                    target_objectnesses[target_argmax],
                )

        self.counter += 1
        return self.total_loss

    def latest_loss_to_tqdm(self) -> str:
        return f"Total: {self.total_loss:.5f} | " + " | ".join(
            [f"{k}: {v:.5f}" for k, v in self.losses.items()]
        )


class ExperimentalLossFailed(torch.nn.Module):

    def __init__(self, dataset: mnn_ordinal.COCOInstances2017Ordinal3):
        self.dataset = dataset
        super().__init__()
        self.xc_loss = torch.nn.BCELoss()
        self.yc_loss = torch.nn.BCELoss()
        self.w_loss = torch.nn.BCELoss()
        self.h_loss = torch.nn.BCELoss()
        self.class_loss = torch.nn.BCELoss()
        self.objectness_loss = torch.nn.BCELoss()
        self.losses = {
            "xc": 0,
            "yc": 0,
            "w": 0,
            "h": 0,
            "class": 0,
            "objectness": 0,
        }
        self.total_loss = 0

    def _total_loss(self):
        return sum(self.losses.values())

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        self.losses = {
            "xc": 0,
            "yc": 0,
            "w": 0,
            "h": 0,
            "class": 0,
            "objectness": 0,
        }
        for pred_sample, target_sample in zip(predictions, targets):
            (
                pred_xc_ordinals,
                pred_yc_ordinals,
                pred_w_ordinals,
                pred_h_ordinals,
                pred_class_scores,
                pred_objectnesses,
            ) = self.dataset.split_output_to_vectors(pred_sample)

            (
                target_xc_ordinals,
                target_yc_ordinals,
                target_w_ordinals,
                target_h_ordinals,
                target_class_scores,
                target_objectnesses,
            ) = self.dataset.split_output_to_vectors(target_sample)

            self.objectness_loss_value = self.objectness_loss(
                pred_objectnesses, target_objectnesses
            )
            self.losses["objectness"] += self.objectness_loss_value

            if target_objectnesses.sum() == 0:
                continue
            vectors_positions = target_objectnesses == 1

            self.xc_loss_value = self.xc_loss(
                pred_xc_ordinals[vectors_positions],
                target_xc_ordinals[vectors_positions],
            )
            self.losses["xc"] += self.xc_loss_value

            self.yc_loss_value = self.yc_loss(
                pred_yc_ordinals[vectors_positions],
                target_yc_ordinals[vectors_positions],
            )
            self.losses["yc"] += self.yc_loss_value

            self.w_loss_value = self.w_loss(
                pred_w_ordinals[vectors_positions], target_w_ordinals[vectors_positions]
            )
            self.losses["w"] += self.w_loss_value

            self.h_loss_value = self.h_loss(
                pred_h_ordinals[vectors_positions], target_h_ordinals[vectors_positions]
            )
            self.losses["h"] += self.h_loss_value

            self.class_loss_value = self.class_loss(
                pred_class_scores[vectors_positions],
                target_class_scores[vectors_positions],
            )
            self.losses["class"] += self.class_loss_value

        self.total_loss = self._total_loss()
        return self.total_loss

    def latest_loss_to_tqdm(self) -> str:
        return f"Total: {self.total_loss:.5f} | " + " | ".join(
            [f"{k}: {v:.5f}" for k, v in self.losses.items()]
        )


class ExperimentalLossTODO(torch.nn.Module):

    def __init__(self, dataset: mnn_ordinal.COCOInstances2017Ordinal3):
        self.dataset = dataset
        super().__init__()
        self.loss_for_raw = FocalLoss(gamma=3)
        self.loss_for_bboxes_focal = FocalLoss(gamma=2)
        self.loss_for_objects_positions_in_grid = FocalLoss(gamma=2)
        self.loss_for_n_objects = torch.nn.MSELoss()
        self.loss_for_bboxes = BboxLoss()
        self.losses = {
            "raw": 0,
            "objects_positions_in_grid": 0,
            "iou": 0,
        }

    def _total_loss(self):
        return sum(self.losses.values())

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        (
            pred_xc_ordinals,
            pred_yc_ordinals,
            pred_w_ordinals,
            pred_h_ordinals,
            pred_class_scores,
            pred_objectnesses,
        ) = self.dataset.split_output_to_vectors_batch(predictions)

        (
            target_xc_ordinals,
            target_yc_ordinals,
            target_w_ordinals,
            target_h_ordinals,
            target_class_scores,
            target_objectnesses,
        ) = self.dataset.split_output_to_vectors_batch(targets)

        # self.objects_positions_in_grid_loss = focal_loss(
        #     pred_objectnesses, target_objectnesses
        # )
        # self.losses["objects_positions_in_grid"] = self.objects_positions_in_grid_loss

        # prediction_n_objects = predictions[:, :, -1].sum(dim=1)
        # target_n_objects = targets[:, :, -1].sum(dim=1)
        # self.n_objects_loss = torch.nn.functional.mse_loss(
        #     prediction_n_objects, target_n_objects
        # )
        # self.losses["n_objects"] = self.n_objects_loss

        predictions_boxes = self.dataset.output_to_bboxes_as_mask_batch(predictions)
        targets_boxes = self.dataset.output_to_bboxes_as_mask_batch(targets)
        self.bboxes_loss = self.loss_for_bboxes(predictions_boxes, targets_boxes)
        self.losses["iou"] = self.bboxes_loss

        # Whole output loss
        # self.raw_loss = self.loss_for_raw(predictions, targets)
        # self.losses["raw"] = self.raw_loss

        return self._total_loss()

    def latest_loss_to_tqdm(self) -> str:
        return " | ".join([f"{k}: {v:.4f}" for k, v in self.losses.items()])
