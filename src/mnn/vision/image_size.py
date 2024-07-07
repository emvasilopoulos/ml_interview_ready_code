import dataclasses


@dataclasses.dataclass
class ImageSize:
    width: int = 640
    height: int = 384
    channels: int = 3

    @staticmethod
    def from_dict(image_size: dict) -> "ImageSize":
        return ImageSize(
            image_size["width"],
            image_size["height"],
            image_size["channels"],
        )
