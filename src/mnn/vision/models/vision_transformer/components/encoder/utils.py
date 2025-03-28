import torch
import mnn.vision.models.vision_transformer.components.encoder.config as mnn_encoder_config

TRANSFORMER_ACTIVATIONS = {
    "relu": torch.nn.ReLU(),
    "gelu": torch.nn.GELU(),
}

ALL_ACTIVATIONS = {
    "relu": torch.nn.ReLU(),
    "gelu": torch.nn.GELU(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "identity": torch.nn.Identity(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "elu": torch.nn.ELU(),
    "selu": torch.nn.SELU(),
    "softplus": torch.nn.Softplus(),
    "softshrink": torch.nn.Softshrink(),
    "softsign": torch.nn.Softsign(),
    "hardshrink": torch.nn.Hardshrink(),
    "tanhshrink": torch.nn.Tanhshrink(),
    "hardtanh": torch.nn.Hardtanh(),
    "log_sigmoid": torch.nn.LogSigmoid(),
    "softmin": torch.nn.Softmin(),
    "softmax": torch.nn.Softmax(),
    "log_softmax": torch.nn.LogSoftmax(),
    "hardsigmoid": torch.nn.Hardsigmoid(),
    "hardswish": torch.nn.Hardswish(),
    "mish": torch.nn.Mish(),
    "silu": torch.nn.SiLU(),
    "celu": torch.nn.CELU(),
    "glu": torch.nn.GLU(),
}


def get_activation_from_config(
    transformer_encoder_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
):
    activation_name = transformer_encoder_config.activation
    return TRANSFORMER_ACTIVATIONS[activation_name]


def get_combinator_activation_from_config(
    transformer_encoder_config: mnn_encoder_config.CombinatorConfiguration,
):
    activation_name = transformer_encoder_config.combinator_activation
    return ALL_ACTIVATIONS[activation_name]


def get_transformer_encoder_from_config(
    transformer_encoder_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
):
    d_model = transformer_encoder_config.d_model
    n_head = transformer_encoder_config.n_heads
    ff_dim = transformer_encoder_config.feed_forward_dimensions
    activation_name = transformer_encoder_config.activation
    activation = TRANSFORMER_ACTIVATIONS[activation_name]

    """
    # TODO - read https://arxiv.org/pdf/2002.04745v1.pdf to decide if norm_first should be True or False
    # Also, PyTorch vit does norm_first = True https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py 
    """
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=ff_dim,
        activation=activation,
        batch_first=True,
        dropout=0.1,  # Default
        norm_first=True,
    )
    num_layers = transformer_encoder_config.number_of_layers

    if transformer_encoder_config.layer_norm_config.has_layer_norm:
        layer_norm_eps = transformer_encoder_config.layer_norm_config.eps
        bias = transformer_encoder_config.layer_norm_config.bias
        encoder_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)

        return torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm,
            mask_check=transformer_encoder_config.mask_check,
            enable_nested_tensor=False,  # Using ReLU or GELU so either way it is not using nested tensor
        )
    else:
        return torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            mask_check=transformer_encoder_config.mask_check,
            enable_nested_tensor=False,  # Using ReLU or GELU so either way it is not using nested tensor
        )


if __name__ == "__main__":

    # I can provide any size of image as long as its dimensions are divisible by patch_size

    # image to pytorch tensor
    batch_size = 1
    sequence_length = 240
    feature_vector_size = 768
    my_input = torch.randn(batch_size, sequence_length, feature_vector_size)

    encoder_config = mnn_encoder_config.VisionTransformerEncoderConfiguration(
        use_cnn=False,
        d_model=feature_vector_size,
        n_heads=16,
        feed_forward_dimensions=512,
        number_of_layers=1,
        eps=2e-5,
    )

    my_encoder = get_transformer_encoder_from_config(encoder_config)
    my_output = my_encoder(my_input)
    print("--------- Encoder ---------")
    print(my_encoder)
    print("--------- Output ---------")
    print(my_output.shape)
