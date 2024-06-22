import abc
import pathlib
from typing import Any, Dict, List
import cv2
import torch.utils.data


class BaseCelebaDataset(torch.utils.data.Dataset):
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

        self.annotations_path = dataset_path / self.__get_annotations_filename()
        self.annotations = self.__parse_annotations()

    @abc.abstractmethod
    def __get_annotations_filename(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def __parse_annotations(self) -> Dict[str, List[int]]:
        raise NotImplementedError()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index) -> Any:
        image_path = self.images_paths[index]
        image = self.__load_image(image_path)
        annotation = self.annotations[image_path.name]

    def __load_image(self, image_path: pathlib.Path) -> torch.Tensor:
        return torch.tensor(cv2.imread(image_path.as_posix())).permute(2, 0, 1).float()
