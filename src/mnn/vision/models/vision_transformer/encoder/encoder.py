import dataclasses

import torch
import torch.nn

from mnn.vision.models.vision_transformer.patcher import (
    PatchingLayer,
)
import mnn.vision.models.vision_transformer.embedders.fully_connected as mnn_fc
import mnn.vision.models.vision_transformer.encoder.config as mnn_config
import mnn.vision.models.vision_transformer.encoder.utils as mnn_encoder_utils
import mnn.vision.models.vision_transformer.positional_encoders.sinusoidal as mnn_sinusoidal_positional_encoders

# Vision Transformer Implementation
# https://arxiv.org/pdf/2010.11929
# https://arxiv.org/pdf/2106.14881 # improvements with CNNs


@dataclasses.dataclass
class VisionTranformerImageSize:
    width: int = 640
    height: int = 384
    channels: int = 3


def transformer_sequence_length(patch_size: int, image_size: VisionTranformerImageSize):
    return (image_size.width // patch_size) * (image_size.height // patch_size)


class VisionTransformerEncoder(torch.nn.Module):
    patch_size: int = 32
    patcher: PatchingLayer
    patch_embedder: mnn_fc.FullyConnectedPatchEmbedder
    position_embedder: mnn_sinusoidal_positional_encoders.PositionalEncoding
    encoder_layer: torch.nn.TransformerEncoderLayer
    encoder: torch.nn.TransformerEncoder

    def __init__(
        self,
        transformer_encoder_config: mnn_config.VisionTransformerEncoderConfiguration,
        input_image_size: VisionTranformerImageSize,
    ):
        super().__init__()

        self.input_image_size = input_image_size
        self.transformer_encoder_config = transformer_encoder_config

        if transformer_encoder_config.use_cnn:
            # Simple CNN to extract patches
            self.patcher = torch.nn.Conv2d(
                in_channels=3,
                out_channels=hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            ## TODO - Implement more complex CNNs
        else:
            self.patcher = PatchingLayer(
                patch_size=self.patch_size,
                image_height=self.input_image_size.height,
                image_width=self.input_image_size.width,
            )
            self.patch_embedder = mnn_fc.FullyConnectedPatchEmbedder(
                full_patch_size=self.patch_size * self.patch_size * 3,
                transformer_hidden_dimension_size=transformer_encoder_config.hidden_dim,
            )

        hidden_dim = transformer_encoder_config.hidden_dim
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.sequence_length = transformer_sequence_length(
            self.patch_size, input_image_size
        )
        self.sequence_length += 1  # Add class token - Don't know why. BERT does it and the paper mentions it

        self.positional_encoder = mnn_sinusoidal_positional_encoders.PositionalEncoding(
            number_of_tokens=self.sequence_length, size_of_token_embedding=hidden_dim
        )

        self.encoder = mnn_encoder_utils.get_transformer_encoder_from_config(
            transformer_encoder_config
        )

    def prepend_xclass_token(
        self, batch_size: int, embeddings: torch.Tensor
    ) -> torch.Tensor:
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(batch_size, -1, -1)
        sequence_dimension_index = 1
        return torch.cat([batch_class_token, embeddings], dim=sequence_dimension_index)

    def interpolate_positional_embeddings(
        self, positional_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Reason --> inject inductive bias about 2D structure of the image
        """
        raise NotImplementedError()

    def forward(self, images_batch: torch.Tensor) -> torch.Tensor:
        # Step 1 - Split image(s) into fixed-size patches
        patches_batch = self.patcher(images_batch)
        patches_batch = patches_batch.flatten(start_dim=2)
        # The self attention layer expects inputs in the format (batch_size, seq_length, embedding_size)
        # Step 2 - flatten patches and map each to embeddings of D dimension
        embeddings_batch = self.patch_embedder(
            patches_batch
        )  # Referred as patch embeddings in the paper
        # Step 3 - Prepend a learnable embedding to the patch embeddings
        embeddings_batch = self.prepend_xclass_token(
            batch_size=embeddings_batch.shape[0], embeddings=embeddings_batch
        )

        # Step 4 - Add positional embeddings to retain positional information
        # is this step taken care of by the transformer encoder?
        # if not, find out the next comment's implementation
        # positional_embeddings_batch = self.position_embedder(embeddings_batch)
        # interpolated_embeddings_batch = self.interpolate_positional_embeddings(
        #     positional_embeddings_batch
        # )
        positioned_embeddings_batch = self.positional_encoder(embeddings_batch)
        # Step 5 - Pass positional embeddings through the transformer encoder
        transformer_encodings_batch = self.encoder(positioned_embeddings_batch)
        # investigate the following from source code of torchvision.VisionTransformer
        """
        The authors of the vision transformer paper state:
        "...we prepend a learnable embedding to the sequence of embed- ded patches (z0 = xclass),
        whose state at the output of the Transformer encoder (z0L) serves as the image representation y"
        """
        # y = transformer_encodings_batch[
        #     :, 0
        # ]  # so this is the class token after passing through the encoder

        return transformer_encodings_batch


"""
NOTES
1. It is often beneficial to fine-tune at higher resolution than pre-training

"""

""" HELP WITH IMPLEMENTATION """
# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

if __name__ == "__main__":
    import time

    n = 1
    patch_size = 32
    hidden_dim = 384
    import cv2

    image = cv2.imread("../../../../../../data/image/alan_resized.jpeg")
    image_size = VisionTranformerImageSize(
        height=image.shape[0], width=image.shape[1], channels=image.shape[2]
    )
    # image to pytorch tensor
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    image = torch.randn(n, image_size.channels, image_size.height, image_size.width)

    encoder_config = mnn_config.VisionTransformerEncoderConfiguration(
        use_cnn=False, d_model=hidden_dim, hidden_dim=hidden_dim
    )
    my_encoder = VisionTransformerEncoder(encoder_config, image_size)
    # print(my_encoder.encoder)

    output = my_encoder(image)
    # revert to opencv format
    output = output[0]
    image_output = cv2.cvtColor(output.detach().numpy(), cv2.COLOR_GRAY2BGR)
    cv2.imshow("output", image_output)
    cv2.waitKey(0)
    # patch_layer = PatchingLayer(patch_size)

    # cnn_patcher = torch.nn.Conv2d(
    #     in_channels=3,
    #     out_channels=hidden_dim,
    #     kernel_size=patch_size,
    #     stride=patch_size,
    # )

    # t0 = time.time()
    # patches = patch_layer(image)
    # t1 = time.time()
    # print(f"patch layer:", t1 - t0, patches.shape)

    # t0 = time.time()
    # cnn_patches = cnn_patcher(image)
    # t1 = time.time()
    # print(f"cnn patcher:", t1 - t0, cnn_patches.shape)
    # n_h = image_size.height // patch_size
    # n_w = image_size.width // patch_size
    # x = cnn_patches.reshape(n, hidden_dim, n_h * n_w)
    # print("Reshaped:", x.shape)
    # x = x.permute(0, 2, 1)
    # print("Permuted:", x.shape)
