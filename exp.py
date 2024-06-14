import torch
import numpy as np
from PIL import Image

from mnn.vision.preprocess.pil import PILPreprocessor
from mnn.vision.models.example import ExampleNet


def tensor_image_analysis(tensor_image):
    print(tensor_image.shape)
    print(torch.min(tensor_image))
    print(torch.max(tensor_image))


if __name__ == "__main__":
    random_image = np.random.randint(
        low=0, high=256, dtype=np.uint8, size=(400, 400, 3)
    )
    device = torch.device("cpu")
    tensor_random_image = PILPreprocessor.to_tensor(
        image=Image.fromarray(random_image), device=device
    ).unsqueeze(0)
    model = ExampleNet(*random_image.shape, n_classes=3)
    model.eval()
    model.to(device)
    print(model(tensor_random_image))
