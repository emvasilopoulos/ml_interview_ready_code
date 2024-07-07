import dataclasses

import mnn.vision.image_size


@dataclasses.dataclass
class HyperparametersConfiguration:
    image_size: mnn.vision.image_size.ImageSize
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
