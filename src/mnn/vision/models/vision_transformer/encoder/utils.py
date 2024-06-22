import torch
import mnn.vision.models.vision_transformer.encoder.config as mnn_config


def get_transformer_encoder_from_config(
    transformer_encoder_config: mnn_config.VisionTransformerEncoderConfiguration,
):
    d_model = transformer_encoder_config.d_model
    n_head = transformer_encoder_config.n_heads
    ff_dim = transformer_encoder_config.feed_forward_dimensions
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=ff_dim,
        activation="gelu",
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
    )
