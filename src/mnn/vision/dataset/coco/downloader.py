import pathlib
import sys
import urllib.request

BASE_URL = "http://images.cocodataset.org/"


def _bytes_to_mbytes(bytes: int) -> float:
    return bytes / 1024**2


def report_progress(block_num, block_size, total_size):
    downloaded_size = block_num * block_size
    if total_size > 0:
        percent = downloaded_size * 100 / total_size
        sys.stdout.write(
            f"\rDownloaded {_bytes_to_mbytes(downloaded_size):.2f}MB of {_bytes_to_mbytes(total_size):.2f}MB ({percent:.2f}%)"
        )
        sys.stdout.flush()
    if downloaded_size >= total_size:
        print()  # New line after completion


def _download_zip_file(url: str, save_dir: pathlib.Path, zip_file: str):
    urllib.request.urlretrieve(
        url, (save_dir / zip_file).as_posix(), reporthook=report_progress
    )


def download_train_val_annotations_2017(save_dir: pathlib.Path):
    zip_file = "annotations_trainval2017.zip"
    url = BASE_URL + "annotations/" + zip_file
    _download_zip_file(url, save_dir, zip_file)


def download_train_images_2017(save_dir: pathlib.Path):
    zip_file = "train2017.zip"
    url = BASE_URL + "zips/" + zip_file
    _download_zip_file(url, save_dir, zip_file)


def download_val_images_2017(save_dir: pathlib.Path):
    zip_file = "val2017.zip"
    url = BASE_URL + "zips/" + zip_file
    _download_zip_file(url, save_dir, zip_file)


def download_all_coco_2017(save_dir: pathlib.Path):
    download_train_val_annotations_2017(save_dir)
    download_train_images_2017(save_dir)
    download_val_images_2017(save_dir)
