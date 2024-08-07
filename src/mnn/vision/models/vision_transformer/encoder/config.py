import dataclasses
from typing import Any, Dict

import torch


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
    n_heads: int = 16  #
    # the activation function in the feedforward network model
    activation: str = "gelu"
    # Alterantive name --> hidden_dim
    feed_forward_dimensions: int = 512  # the dimension of the feedforward network model
    # the eps value in final LayerNorm
    eps: int = 1e-5
    # whether to use bias in layernorm components
    bias: bool = True
    # whether to check for mask in forward pass
    mask_check: bool = False

    @staticmethod
    def from_dict(
        model_configuration: Dict[str, Any]
    ) -> "VisionTransformerEncoderConfiguration":
        return VisionTransformerEncoderConfiguration(
            model_configuration["has_positional_encoding"],
            model_configuration["is_input_to_positional_encoder_normalized"],
            model_configuration["number_of_layers"],
            model_configuration["d_model"],
            model_configuration["n_heads"],
            model_configuration["activation"],
            model_configuration["feed_forward_dimensions"],
            float(model_configuration["eps"]),
            model_configuration["bias"],
            model_configuration["mask_check"],
        )


FLOATING_POINT_PRECISIONS = {"float32": torch.float32, "float16": torch.float16}


@dataclasses.dataclass
class MyVisionTransformerConfiguration:
    dtype: torch.dtype
    rgb_combinator_config: VisionTransformerEncoderConfiguration
    encoder_config: VisionTransformerEncoderConfiguration

    @staticmethod
    def from_dict(
        model_configuration: Dict[str, Any]
    ) -> "MyVisionTransformerConfiguration":
        if not model_configuration["dtype"] in FLOATING_POINT_PRECISIONS:
            raise ValueError(
                f"Unsupported dtype {model_configuration['dtype']}. Supported dtypes are {FLOATING_POINT_PRECISIONS.keys()}"
            )
        return MyVisionTransformerConfiguration(
            dtype=FLOATING_POINT_PRECISIONS[model_configuration["dtype"]],
            rgb_combinator_config=VisionTransformerEncoderConfiguration.from_dict(
                model_configuration["RGBCombinator"]
            ),
            encoder_config=VisionTransformerEncoderConfiguration.from_dict(
                model_configuration["VisionTransformerEncoder"]
            ),
        )
