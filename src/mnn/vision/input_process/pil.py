import torch
from torchvision import transforms
from PIL import Image

# ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# Otherwise calculate mean & std to your own dataset
# Otherwise just use mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]


class PILPreprocessor:
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    @classmethod
    def to_tensor(cls, image: Image, device: torch.device) -> torch.tensor:
        return cls.transform(image).to(device)

    @classmethod
    def image_path_to_tensor_torchvision_way(
        cls, image_path: str, device: torch.device
    ):
        return cls.to_tensor(Image.open(image_path), device=device)
