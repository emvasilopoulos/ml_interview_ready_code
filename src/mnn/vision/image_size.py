import dataclasses


@dataclasses.dataclass
class ImageSize:
    width: int = 640
    height: int = 384
    channels: int = 3
