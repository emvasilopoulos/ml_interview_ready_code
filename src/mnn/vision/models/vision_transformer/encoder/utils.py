import torch
import mnn.vision.models.vision_transformer.encoder.config as mnn_encoder_config

activation_functions = {
    "sigmoid": torch.nn.Sigmoid(),
    "relu": torch.nn.ReLU(),
    "gelu": torch.nn.GELU(),
}


def get_transformer_encoder_from_config(
    transformer_encoder_config: mnn_encoder_config.VisionTransformerEncoderConfiguration,
):
    d_model = transformer_encoder_config.d_model
    n_head = transformer_encoder_config.n_heads
    ff_dim = transformer_encoder_config.feed_forward_dimensions
    activation_name = transformer_encoder_config.activation
    activation = activation_functions[activation_name]

    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=ff_dim,
        activation=activation,
        batch_first=True,
    )

    layer_norm_eps = transformer_encoder_config.eps
    bias = transformer_encoder_config.bias
    encoder_norm = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)

    num_layers = transformer_encoder_config.number_of_layers
    return torch.nn.TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=num_layers,
        norm=encoder_norm,
        mask_check=transformer_encoder_config.mask_check,
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
