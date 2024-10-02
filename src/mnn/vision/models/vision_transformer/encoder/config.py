import dataclasses
from typing import Any, Dict


@dataclasses.dataclass
class LayerNormConfiguration:
    has_layer_norm: bool
    eps: int
    elementwise_affine: bool
    bias: bool

    @staticmethod
    def from_dict(model_configuration: Dict[str, Any]) -> "LayerNormConfiguration":
        return LayerNormConfiguration(
            has_layer_norm=model_configuration["has_layer_norm"],
            eps=float(model_configuration["eps"]),
            elementwise_affine=model_configuration["elementwise_affine"],
            bias=model_configuration["bias"],
        )


@dataclasses.dataclass
class VisionTransformerEncoderConfiguration:
    has_positional_encoding: bool
    is_input_to_positional_encoder_normalized: bool
    # the number of transformer encoder layers in the encoder
    number_of_layers: int
    """
    The number of expected features in the encoder/decoder inputs.

    In the vision transformer paper it is stated:
    "The Transformer uses constant latent vector size D through all of its layers, so
    we flatten the patches and map to D dimensions with trainable linear projections."
    """
    d_model: int
    """
    The number of heads in the multi-head attention models

    Multi-head attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    """
    n_heads: int
    # the activation function in the feedforward network model
    activation: str
    # Alterantive name --> hidden_dim
    feed_forward_dimensions: int  # the dimension of the feedforward network model

    layer_norm_config: LayerNormConfiguration

    # whether to check for mask in forward pass
    mask_check: bool = False

    @staticmethod
    def from_dict(
        model_configuration: Dict[str, Any]
    ) -> "VisionTransformerEncoderConfiguration":
        return VisionTransformerEncoderConfiguration(
            has_positional_encoding=model_configuration["has_positional_encoding"],
            is_input_to_positional_encoder_normalized=model_configuration[
                "is_input_to_positional_encoder_normalized"
            ],
            number_of_layers=model_configuration["number_of_layers"],
            d_model=model_configuration["d_model"],
            n_heads=model_configuration["n_heads"],
            activation=model_configuration["activation"],
            feed_forward_dimensions=model_configuration["feed_forward_dimensions"],
            layer_norm_config=LayerNormConfiguration.from_dict(
                model_configuration["LayerNorm"]
            ),
            mask_check=model_configuration["mask_check"],
        )


@dataclasses.dataclass
class MyBackboneVitConfiguration:
    rgb_combinator_config: VisionTransformerEncoderConfiguration
    encoder_config: VisionTransformerEncoderConfiguration

    @staticmethod
    def from_dict(model_configuration: Dict[str, Any]) -> "MyBackboneVitConfiguration":
        return MyBackboneVitConfiguration(
            rgb_combinator_config=VisionTransformerEncoderConfiguration.from_dict(
                model_configuration["RGBCombinator"]
            ),
            encoder_config=VisionTransformerEncoderConfiguration.from_dict(
                model_configuration["VisionTransformerEncoder"]
            ),
        )
