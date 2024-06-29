import abc
import pathlib
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch.utils.data

import mnn.vision.dataset.utilities as utilities


def _annotation_as_tensor(annotation: List[int]) -> torch.Tensor:
    return torch.tensor(annotation).float()


def _clean_line(line: str) -> str:
    return [x for x in line.strip().split(" ") if x != ""]


class BaseCelebaDataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    def __init__(self, dataset_path: pathlib.Path, use_mini: bool = True) -> None:
        """
        Dataset structure:
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

    @abc.abstractmethod
    def _preprocess_annotation(
        self, annotation: torch.Tensor, image_tensor: torch.Tensor = None
    ) -> Any:
        raise NotImplementedError()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images_paths[index]
        image = utilities.load_image_as_tensor(image_path)
        annotation = _annotation_as_tensor(self.annotations[image_path.name])
        return image, self._preprocess_annotation(annotation, image)


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

    def _preprocess_annotation(
        self, annotation: torch.Tensor, image_tensor: torch.Tensor = None
    ) -> Any:
        return annotation


class CelebaAttributesDataset(BaseCelebaDataset):
    def _get_annotations_filename(self) -> str:
        return "list_attr_celeba.txt"

    def _set_attributes(self, attributes: List[str]) -> None:
        """
        Values for each attribute are:
            -1: Not present
            1: Present
        Args:
            attributes (List[str]): 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young
        """
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

    def _preprocess_annotation(
        self, annotation: torch.Tensor, image_tensor: torch.Tensor = None
    ) -> Any:
        return torch.where(annotation == -1, torch.tensor(0), annotation)


class CelebaBBoxDataset(BaseCelebaDataset):
    def _get_annotations_filename(self) -> str:
        return "list_bbox_celeba.txt"

    def _set_attributes(self, attributes: List[str]) -> None:
        """

        Args:
            attributes (List[str]): image_id x_1 y_1 width height
        """
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

    def _preprocess_annotation(
        self, annotation: torch.Tensor, image_tensor: torch.Tensor = None
    ) -> Any:
        annotation[0] /= image_tensor.shape[2]  # x_1
        annotation[1] /= image_tensor.shape[1]  # y_1
        annotation[2] /= image_tensor.shape[2]  # width
        annotation[3] /= image_tensor.shape[1]  # height
        return annotation


class CelebaAlignedLandmarksDataset(BaseCelebaDataset):
    def _get_annotations_filename(self) -> str:
        return "list_landmarks_align_celeba.txt"

    def _set_attributes(self, attributes: List[str]) -> None:
        """
        Args:
            attributes (List[str]): lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
        """
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

    def _preprocess_annotation(
        self, annotation: torch.Tensor, image_tensor: torch.Tensor = None
    ) -> Any:
        annotation[::2] /= image_tensor.shape[2]  # even indices are x
        annotation[1::2] /= image_tensor.shape[1]  # odd indices are y
        return annotation


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

    def _preprocess_annotation(
        self, annotation: torch.Tensor, image_tensor: torch.Tensor = None
    ) -> Any:
        annotation[::2] /= image_tensor.shape[2]  # even indices are x
        annotation[1::2] /= image_tensor.shape[1]  # odd indices are y
        return annotation


if __name__ == "__main__":
    celeba = CelebaBBoxDataset(
        pathlib.Path("/Users/emlvasilopoulos/Datasets/CelebA"), use_mini=True
    )
    img_tensor, annotation_tensor = celeba[1]
    print(img_tensor.shape)
    print(celeba.attributes)
    print(annotation_tensor)
