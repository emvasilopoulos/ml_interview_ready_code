import torch


def pad_image(
    x: torch.Tensor,
    pad_dimension: int,
    expected_dimension_size: int,
    padding_percent: float,
    pad_value: int = 0,
) -> torch.Tensor:
    if padding_percent < 0 or padding_percent > 1:
        raise ValueError("The padding_percent should be between 0 and 1")

    x_dim_size = x.shape[pad_dimension]
    pad_amount = expected_dimension_size - x_dim_size
    top_pad = int(pad_amount * padding_percent)
    bottom_pad = pad_amount - top_pad
    if pad_dimension == 1:
        pad_amount_tuple = (0, 0, top_pad, bottom_pad)
    elif pad_dimension == 2:
        pad_amount_tuple = (top_pad, bottom_pad, 0, 0)
    else:
        raise ValueError("The pad_dimension should be 1 or 2")
    if pad_amount < 0:
        raise ValueError(
            f"The image is already bigger than the expected height for incoming image: {x.shape}"
        )
    return torch.nn.functional.pad(
        x, pad_amount_tuple, mode="constant", value=pad_value
    )
