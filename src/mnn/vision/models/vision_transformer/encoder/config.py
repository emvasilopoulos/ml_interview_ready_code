import dataclasses


@dataclasses.dataclass
class VisionTransformerEncoderConfiguration:
    use_cnn: bool
    patch_size: int = 32
    number_of_layers: int = 1  # the number of sub-encoder-layers in the encoder

    # the number of expected features in the encoder/decoder inputs
    """
    In the vision transformer paper it is stated:
    "The Transformer uses constant latent vector size D through all of its layers, so
    we flatten the patches and map to D dimensions with trainable linear projections."
    """
    d_model: int = 384

    n_heads: int = 16  # the number of heads in the multiheadattention models

    # Alterantive name --> hidden_dim
    feed_forward_dimensions: int = 512  # the dimension of the feedforward network model

    eps = 1e-5  # the eps value in layer normalization components
    bias = True  # whether to use bias in layernorm components
