import abc
import pathlib
from typing import Any, Dict, List, Tuple
import cv2
import torch.utils.data


def __read_image(path: pathlib.Path) -> torch.Tensor:
    return cv2.imread(path.as_posix())


def _clean_line(line: str) -> str:
    return [x for x in line.strip().split(" ") if x != ""]


def _load_image_as_tensor(image_path: pathlib.Path) -> torch.Tensor:
    return torch.tensor(__read_image(image_path)).permute(2, 0, 1).float()


def _annotation_as_tensor(annotation: List[int]) -> torch.Tensor:
    return torch.tensor(annotation).float()


class BaseCelebaDataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    def __init__(self, dataset_path: pathlib.Path, use_mini: bool = True) -> None:
        """
        Dataset structur:
        /path/to/dataset/
            img_align_celeba/
                000001.jpg
                000002.jpg
                ...
            list_attr_celeba.txt
            list_bbox_celeba.txt
            list_eval_partition.txt
            list_landmarks_align_celeba.txt
            list_landmarks_celeba.txt
            identity_CelebA.txt

        Args:
            dataset_path (str): Path to the CelebA dataset directory.
            use_mini (bool): Whether to use the mini version of the dataset. Useful for development and testing.
        """
        super().__init__()

        if not dataset_path.exists():
            raise FileNotFoundError(f"Path: '{dataset_path}' does not exist...")
        if use_mini:
            images_subdir = "mini"  # contains 20% of the original dataset
        else:
            images_subdir = "img_align_celeba"
        self.images_path = dataset_path / images_subdir
        self.images_paths = list(self.images_path.glob("*.jpg"))

        self.annotations_path = dataset_path / self._get_annotations_filename()
        self.annotations = self._parse_annotations(self.annotations_path)

    @abc.abstractmethod
    def _get_annotations_filename(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def _parse_annotations(
        self, annotations_path: pathlib.Path
    ) -> Dict[str, List[int]]:
        """
        all CelebA datasets have the same structure following the formula:
        image_name attribute_1 attribute_2 ... attribute_n
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images_paths[index]
        image = _load_image_as_tensor(image_path)
        annotation = _annotation_as_tensor(self.annotations[image_path.name])
        return image, annotation


class CelebaIdentityDataset(BaseCelebaDataset):
    def _get_annotations_filename(self) -> str:
        return "identity_CelebA.txt"

    def _parse_annotations(
        self, annotations_path: pathlib.Path
    ) -> Dict[str, List[int]]:
        annotations = {}
        with annotations_path.open(mode="r") as f:
            for line in f:
                line_elements = _clean_line(line)
                image_name, identity = line_elements[0], line_elements[1]
                annotations[image_name] = [int(identity)]
        return annotations


class CelebaAttributesDataset(BaseCelebaDataset):
    def _get_annotations_filename(self) -> str:
        return "list_attr_celeba.txt"

    def _set_attributes(self, attributes: List[str]) -> None:
        self.attributes = attributes

    def _parse_annotations(
        self, annotations_path: pathlib.Path
    ) -> Dict[str, List[int]]:
        annotations = {}
        with annotations_path.open(mode="r") as f:
            for i, line in enumerate(f):
                if i == 1:
                    self._set_attributes(_clean_line(line))
                elif i > 1:
                    line_elements = _clean_line(line)
                    image_name = line_elements[0]
                    attributes_values = [int(x) for x in line_elements[1:]]
                    annotations[image_name] = attributes_values
        return annotations


class CelebaBBoxDataset(BaseCelebaDataset):
    def _get_annotations_filename(self) -> str:
        return "list_bbox_celeba.txt"

    def _set_attributes(self, attributes: List[str]) -> None:
        self.attributes = attributes

    def _parse_annotations(
        self, annotations_path: pathlib.Path
    ) -> Dict[str, List[int]]:
        annotations = {}
        with annotations_path.open(mode="r") as f:
            for i, line in enumerate(f):
                if i == 1:
                    self._set_attributes(_clean_line(line))
                if i > 1:
                    line_elements = _clean_line(line)
                    image_name = line_elements[0]
                    bbox = [int(x) for x in line_elements[1:]]
                    annotations[image_name] = bbox
        return annotations


class CelebaAlignedLandmarksDataset(BaseCelebaDataset):
    def _get_annotations_filename(self) -> str:
        return "list_landmarks_align_celeba.txt"

    def _set_attributes(self, attributes: List[str]) -> None:
        self.attributes = attributes

    def _parse_annotations(
        self, annotations_path: pathlib.Path
    ) -> Dict[str, List[int]]:
        annotations = {}
        with annotations_path.open(mode="r") as f:
            for i, line in enumerate(f):
                if i == 1:
                    self._set_attributes(_clean_line(line))
                if i > 1:
                    line_elements = _clean_line(line)
                    image_name = line_elements[0]
                    landmarks = [int(x) for x in line_elements[1:]]
                    annotations[image_name] = landmarks
        return annotations


class CelebaLandmarksDataset(BaseCelebaDataset):
    def _get_annotations_filename(self) -> str:
        return "list_landmarks_celeba.txt"

    def _set_attributes(self, attributes: List[str]) -> None:
        self.attributes = attributes

    def _parse_annotations(
        self, annotations_path: pathlib.Path
    ) -> Dict[str, List[int]]:
        annotations = {}
        with annotations_path.open(mode="r") as f:
            for i, line in enumerate(f):
                if i == 1:
                    self._set_attributes(_clean_line(line))
                if i > 1:
                    line_elements = _clean_line(line)
                    image_name = line_elements[0]
                    landmarks = [int(x) for x in line_elements[1:]]
                    annotations[image_name] = landmarks
        return annotations


if __name__ == "__main__":
    celeba = CelebaBBoxDataset(
        pathlib.Path("/Users/emlvasilopoulos/Datasets/CelebA"), use_mini=True
    )
    img_tensor, annotation_tensor = celeba[1]
    print(celeba.attributes)
    print(annotation_tensor)
