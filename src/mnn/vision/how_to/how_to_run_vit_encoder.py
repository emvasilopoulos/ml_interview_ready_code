import time
import cv2
import torch
import mnn.vision.models.vision_transformer.components.encoder.config as mnn_encoder_config
import mnn.vision.image_size
from mnn.vision.models.vision_transformer.components.encoder.raw_vision_encoder import (
    RawVisionTransformerEncoder,
)

if __name__ == "__main__":

    n = 1
    hidden_dim = 640

    # I can provide any size of image as long as its dimensions are divisible by patch_size

    image_size = mnn.vision.image_size.ImageSize(height=640, width=720, channels=3)
    # image to pytorch tensor
    sequence_length = image_size.height
    embedding_size = image_size.width
    hidden_dim = embedding_size
    image = torch.randn(n, sequence_length, image_size.width)

    encoder_config = [
        mnn_encoder_config.VisionTransformerEncoderConfiguration(
            use_cnn=False, d_model=hidden_dim
        ),
        mnn_encoder_config.VisionTransformerEncoderConfiguration(
            use_cnn=False, d_model=hidden_dim
        ),
    ]

    my_encoder = RawVisionTransformerEncoder(encoder_config[0], image_size)

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
