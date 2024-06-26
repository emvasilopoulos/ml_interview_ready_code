import dataclasses


@dataclasses.dataclass
class VisionTransformerEncoderConfiguration:
    use_cnn: bool
    patch_size: int = 32
    number_of_layers: int = 1  # the number of transformer encoder layers in the encoder

    # the number of expected features in the encoder/decoder inputs
    """
    In the vision transformer paper it is stated:
    "The Transformer uses constant latent vector size D through all of its layers, so
    we flatten the patches and map to D dimensions with trainable linear projections."
    """
    d_model: int = 384

    """
    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """
    n_heads: int = 16  # the number of heads in the multiheadattention models

    # Alterantive name --> hidden_dim
    feed_forward_dimensions: int = 512  # the dimension of the feedforward network model
    eps: int = 1e-5  # the eps value in final LayerNorm
    bias: bool = True  # whether to use bias in layernorm components
    mask_check: bool = False  # whether to check for mask in forward pass
