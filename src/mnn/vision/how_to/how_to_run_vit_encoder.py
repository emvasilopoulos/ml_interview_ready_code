import time
import cv2
import torch
import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils
import mnn.vision.models.vision_transformer.encoder.config as mnn_config
from mnn.vision.models.vision_transformer.vit_encoder import (
    VisionTransformerEncoder,
    VisionTranformerImageSize,
)

if __name__ == "__main__":

    n = 1
    patch_size = 32
    hidden_dim = 384

    image = cv2.imread("../../../../../data/image/alan_resized.jpeg")
    image = cv2.resize(image, (704, 416))

    # I can provide any size of image as long as its dimensions are divisible by patch_size

    image_size = VisionTranformerImageSize(
        height=image.shape[0], width=image.shape[1], channels=image.shape[2]
    )
    # image to pytorch tensor
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    image = torch.randn(n, image_size.channels, image_size.height, image_size.width)

    encoder_config = [
        mnn_config.VisionTransformerEncoderConfiguration(
            use_cnn=False, d_model=hidden_dim
        ),
        mnn_config.VisionTransformerEncoderConfiguration(
            use_cnn=False, d_model=hidden_dim
        ),
    ]

    my_encoder = VisionTransformerEncoder(encoder_config, image_size)

    t0 = time.time()
    output = my_encoder(image)
    t1 = time.time()
    print("Time taken:", t1 - t0, "seconds")
    print(output.shape)
    # revert to opencv format
    output = output[0]
    image_output = cv2.cvtColor(output.detach().numpy(), cv2.COLOR_GRAY2BGR)
    # cv2.imshow("output", image_output)
    # cv2.waitKey(0)
