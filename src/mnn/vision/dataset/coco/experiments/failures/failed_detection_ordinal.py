
class COCOInstances2017Ordinal(BaseCOCOInstances2017Ordinal):

    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
        current_image_size: Optional[mnn.vision.image_size.ImageSize] = None,
    ) -> torch.Tensor:
        output_tensor = torch.zeros(
            (self.expected_image_size.height, self.expected_image_size.width)
        )

        # Add vector with number of objects
        n_objects = len(output_tensor)
        if n_objects > self.max_objects:
            """Keep the 'max_objects' objects with the highest area"""
            # This point will never be reached, unless you're a dork and use an image width of less than 420 pixels.
            raise NotImplementedError(
                "Keeping the 'max_objects' objects with the highest area is not implemented yet"
            )
        output_tensor[0, :] = self._create_number_of_objects_vector(
            n_objects, self.expected_image_size.width
        )

        # Add whole image as a bounding box
        vectors_for_output = []
        for i, annotation in enumerate(annotations):
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]
            area = w_norm * h_norm
            # Skip very small bboxes. Bad annotations
            if area < 0.0004:
                continue
            # Skip very close to image borders bboxes. Bad annotations
            if (
                x1_norm > 0.99
                or y1_norm > 0.99
                or (x1_norm + w_norm) <= 0.01
                or (y1_norm + h_norm) <= 0.01
            ):
                continue

            x1 = x1_norm * fixed_ratio_components.resize_width
            y1 = y1_norm * fixed_ratio_components.resize_height
            w = w_norm * fixed_ratio_components.resize_width
            h = h_norm * fixed_ratio_components.resize_height
            x1, y1, w, h = self.map_bbox_to_padded_image(
                x1, y1, w, h, fixed_ratio_components, padding_percent
            )
            xc = x1 + w / 2
            yc = y1 + h / 2
            new_bbox_norm = [
                xc / self.expected_image_size.width,
                yc / self.expected_image_size.height,
                w / self.expected_image_size.width,
                h / self.expected_image_size.height,
            ]
            vector = self._create_object_vector(
                new_bbox_norm, category, self.expected_image_size.width
            )

            vectors_for_output.append((vector, area))

        # Sort by area and then place in output tensor, so there's at least a logic/order in the predictions
        sorted_vectors_for_output = sorted(vectors_for_output, key=lambda x: x[1])
        for i, (vector, area) in enumerate(sorted_vectors_for_output):
            output_tensor[1 + i, :] = vector
        return output_tensor

    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        n_vectors, vector_size = y.shape
        n_objects_vector = y[0, :]
        n_objects = torch.argmax(n_objects_vector)
        if n_objects > self.TOTAL_POSSIBLE_OBJECTS:
            # Unwanted behaviour by the model. Don't do anything. Keep it in mind.
            pass

        bbox_vector_size = vector_size - (self.N_CLASSES + 1)
        _coord_step = bbox_vector_size // 4

        objects = y[
            1 : self.TOTAL_POSSIBLE_OBJECTS + 1, :
        ]  # The rest should be ignored
        img_h, img_w = self.expected_image_size.height, self.expected_image_size.width

        bboxes = []
        categories = []
        objectness_scores = []
        for o in objects:
            objectness_score = o[-1]
            if filter_by_objectness_score and objectness_score < 0.5:
                continue
            bbox_raw = o[: (len(o) - 80)]
            xc = self._decode_coordinate_vector(bbox_raw[:_coord_step], img_w)
            yc = self._decode_coordinate_vector(
                bbox_raw[_coord_step : 2 * _coord_step], img_h
            )
            w = self._decode_coordinate_vector(
                bbox_raw[2 * _coord_step : 3 * _coord_step], img_w
            )
            h = self._decode_coordinate_vector(
                bbox_raw[3 * _coord_step : 4 * _coord_step], img_h
            )

            bbox = [xc, yc, w, h]
            if all(x == 0 for x in bbox):
                continue
            bboxes.append(bbox)
            category = torch.argmax(o[(len(o) - 80) : -1])
            categories.append(category)
            objectness_scores.append(objectness_score)

        # # Sort by objectness score
        # sorted_indices = sorted(
        #     range(len(objectness_scores)),
        #     key=lambda k: objectness_scores[k],
        #     reverse=True,
        # )
        # bboxes = [bboxes[i] for i in sorted_indices]
        # categories = [categories[i] for i in sorted_indices]
        # objectness_scores = [objectness_scores[i] for i in sorted_indices]
        return bboxes[:n_objects], categories[:n_objects], objectness_scores[:n_objects]


class COCOInstances2017Ordinal2(BaseCOCOInstances2017Ordinal):

    def _make_annotations_to_vectors(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float,
    ):
        vectors_for_output = []
        for i, annotation in enumerate(annotations):
            category = int(annotation["category_id"]) - 1
            if self.desired_classes is not None and not (
                self.desired_classes[0] <= category <= self.desired_classes[-1]
            ):
                continue

            x1_norm, y1_norm, w_norm, h_norm = annotation["normalized_bbox"]
            area = w_norm * h_norm
            # Skip very small bboxes. Bad annotations
            if area < 0.0004:
                continue
            # Skip very close to image borders bboxes. Bad annotations
            if (
                x1_norm > 0.99
                or y1_norm > 0.99
                or (x1_norm + w_norm) <= 0.01
                or (y1_norm + h_norm) <= 0.01
            ):
                continue

            x1 = x1_norm * fixed_ratio_components.resize_width
            y1 = y1_norm * fixed_ratio_components.resize_height
            w = w_norm * fixed_ratio_components.resize_width
            h = h_norm * fixed_ratio_components.resize_height
            x1, y1, w, h = self.map_bbox_to_padded_image(
                x1, y1, w, h, fixed_ratio_components, padding_percent
            )
            xc = x1 + w / 2
            yc = y1 + h / 2

            new_bbox_norm = [
                xc / self.expected_image_size.width,
                yc / self.expected_image_size.height,
                w / self.expected_image_size.width,
                h / self.expected_image_size.height,
            ]

            vector = self._create_object_vector(
                new_bbox_norm, category, self.expected_image_size.width
            )

            vectors_for_output.append((vector, area))
        return vectors_for_output

    def get_output_tensor(
        self,
        annotations: Dict[str, Any],
        fixed_ratio_components: mnn_resize_fixed_ratio.ResizeFixedRatioComponents,
        padding_percent: float = 0,
        current_image_size: Optional[mnn.vision.image_size.ImageSize] = None,
    ) -> torch.Tensor:
        output_tensor_bboxes = torch.zeros(
            (self.expected_image_size.height, self.expected_image_size.width)
        )

        # Add vector with number of objects
        n_objects = len(annotations)
        if n_objects > self.max_objects:
            """Keep the 'max_objects' objects with the highest area"""
            # This point will never be reached, unless you're a dork and use an image width of less than 420 pixels.
            raise NotImplementedError(
                "Keeping the 'max_objects' objects with the highest area is not implemented yet"
            )

        output_tensor_n_objects = torch.zeros(
            (self.expected_image_size.height, self.expected_image_size.width)
        )
        output_tensor_n_objects[0, :] = self._create_number_of_objects_vector(
            n_objects, self.expected_image_size.width
        )

        # Add whole image as a bounding box
        vectors_for_output = self._make_annotations_to_vectors(
            annotations, fixed_ratio_components, padding_percent
        )

        # FAIL
        # Sort by area and then place in output tensor, so there's at least a logic/order in the predictions
        sorted_vectors_for_output = sorted(vectors_for_output, key=lambda x: x[1])
        for i, (vector, area) in enumerate(sorted_vectors_for_output):
            output_tensor_bboxes[i, :] = vector
        return torch.stack([output_tensor_n_objects, output_tensor_bboxes])

    def decode_output_tensor(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        """
        y: torch.Tensor --> Stack [output_tensor_n_objects, output_tensor_bboxes]
        """
        # Separate the two tensors
        n_objects_vector = y[0, 0, :]
        objects = y[1, : self.TOTAL_POSSIBLE_OBJECTS, :]

        # Extract number of objects
        n_objects = min(torch.argmax(n_objects_vector), self.TOTAL_POSSIBLE_OBJECTS)
        if n_objects == 0:
            return [], [], []

        # Extract bboxes
        _coord_step = self.bbox_vector_size // 4
        h, w = self.expected_image_size.height, self.expected_image_size.width
        bboxes = []
        categories = []
        objectness_scores = []
        for i, o in enumerate(objects):
            objectness_score = o[-1]
            if filter_by_objectness_score and objectness_score < 0.5:
                continue
            total_classes = len(self.classes)
            vector_size = len(o)
            idx_bbox = vector_size - (total_classes + 1)
            bbox_raw = o[:idx_bbox]
            xc = self._decode_coordinate_vector(bbox_raw[:_coord_step], w)
            yc = self._decode_coordinate_vector(
                bbox_raw[_coord_step : 2 * _coord_step], h
            )
            w0 = self._decode_coordinate_vector(
                bbox_raw[2 * _coord_step : 3 * _coord_step], w
            )
            h0 = self._decode_coordinate_vector(
                bbox_raw[3 * _coord_step : 4 * _coord_step], h
            )

            if all(x == 0 for x in [xc, yc, w0, h0]):
                continue

            bbox = [xc, yc, w0, h0]
            idx_category = idx_bbox + total_classes
            category = torch.argmax(o[idx_bbox:idx_category])
            bboxes.append(bbox)
            categories.append(category)
            objectness_scores.append(objectness_score)

        return bboxes[:n_objects], categories[:n_objects], objectness_scores[:n_objects]

    def decode_prediction_raw(
        self, y: torch.Tensor, filter_by_objectness_score: bool = False
    ):
        predicted_objects = y[1, :, :]
        # Extract bboxes
        _coord_step = self.bbox_vector_size // 4
        h, w = self.expected_image_size.height, self.expected_image_size.width
        bboxes = []
        categories = []
        objectness_scores = []
        for i, o in enumerate(predicted_objects):
            objectness_score = o[-1]
            if filter_by_objectness_score and objectness_score < 0.5:
                continue
            total_classes = len(self.classes)
            vector_size = len(o)
            idx_bbox = vector_size - (total_classes + 1)
            bbox_raw = o[:idx_bbox]
            xc = self._decode_coordinate_vector(bbox_raw[:_coord_step], w)
            yc = self._decode_coordinate_vector(
                bbox_raw[_coord_step : 2 * _coord_step], h
            )
            w0 = self._decode_coordinate_vector(
                bbox_raw[2 * _coord_step : 3 * _coord_step], w
            )
            h0 = self._decode_coordinate_vector(
                bbox_raw[3 * _coord_step : 4 * _coord_step], h
            )

            if all(x == 0 for x in [xc, yc, w0, h0]):
                continue

            bbox = [xc, yc, w0, h0]
            idx_category = idx_bbox + total_classes
            category = torch.argmax(o[idx_bbox:idx_category])
            bboxes.append(bbox)
            categories.append(category)
            objectness_scores.append(objectness_score)

        return bboxes, categories, objectness_scores


def write_image_with_output(
    temp_out: torch.Tensor,
    validation_image: torch.Tensor,
    sub_dir: str = "any",
):
    bboxes, categories, objectness_scores = (
        COCOInstances2017Ordinal.decode_output_tensor(temp_out.squeeze(0))
    )

    validation_img = validation_image.squeeze(0).detach().cpu()
    validation_img = validation_img.permute(1, 2, 0)
    image = (validation_img.numpy() * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for bbox, category in zip(bboxes, categories):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(category.item()),
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    # reverse mask
    os.makedirs(f"assessment_images/{sub_dir}", exist_ok=True)
    cv2.imwrite(f"assessment_images/{sub_dir}/bboxed_image.jpg", image)




if __name__ == "__main__":
    import pathlib
    import mnn.vision.image_size
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()
    idx = args.idx

    dataset_dir = pathlib.Path(
        "/home/emvasilopoulos/projects/ml_interview_ready_code/data/coco"
    )
    expected_image_size = mnn.vision.image_size.ImageSize(width=640, height=480)

    val_dataset = COCOInstances2017Ordinal(dataset_dir, "val", expected_image_size)
    image_batch, target0 = val_dataset[idx]
    write_image_with_output(
        target0.unsqueeze(0),
        image_batch.unsqueeze(0),
        "train_image_ground_truth",
    )
    image, target = val_dataset.get_pair(idx)
    image = image.permute(1, 2, 0)
    image = (image.numpy() * 255).astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for annotation in target:
        bbox = [int(x) for x in annotation["bbox"]]
        cv2.rectangle(
            image,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (0, 255, 0),
            2,
        )
    cv2.imwrite("assessment_images/train_image_ground_truth/ground_truth.jpg", image)
