from typing import Any, Dict, List
import json
import pathlib


class RawCOCOAnnotationsParser:

    def __init__(self, annotations_path: pathlib.Path):
        self.annotations_path = annotations_path
        self.objects_by_image_id: Dict[str, List[Any]] = {}
        self.data_to_store = {}

    def parse_data(self):
        self._load_json_data()
        self.group_objects_by_image_id()
        self.data_to_store["annotations_grouped_by_image_id"] = self.objects_by_image_id
        self.data_to_store["images"] = self.images

    def write_data(self, output_path: pathlib.Path):
        with open(output_path, "w") as f:
            json.dump(self.data_to_store, f)

    def _load_json_data(self):
        with open(self.annotations_path, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]

    def group_objects_by_image_id(self):
        for annotation in self.annotations:
            image_id = annotation["image_id"]
            if image_id not in self.objects_by_image_id:
                self.objects_by_image_id[image_id] = []
            self.objects_by_image_id[image_id].append(annotation)
        return self.objects_by_image_id
