import numpy as np
import torch
import torch.nn


def get_sinusoid_table(num_tokens: int, token_len: int) -> np.ndarray:
    """Make Sinusoid Table

    Args:
        num_tokens (int): number of tokens
        token_len (int): length of a token

    Returns:
        (torch.FloatTensor) sinusoidal position encoding table
    """

    def get_position_angle_vec(i):
        return [i / np.power(10000, 2 * (j // 2) / token_len) for j in range(token_len)]

    sinusoid_table = np.array([get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table


class PositionalEncoding(torch.nn.Module):
    """
    Functionally, the position embedding is a matrix with the same shape as the tokens
    Tokens have a shape --> (number_of_tokens, size_of_token_embedding)
    """

    def __init__(self, number_of_tokens: int, size_of_token_embedding: int):
        # The paper suggests using 1D positional embeddings
        # 2D positional embeddings do not have any advantages
        self.layer = torch.FloatTensor(
            get_sinusoid_table(number_of_tokens, size_of_token_embedding)
        ).unsqueeze(0)
        pass

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return patches + self.layer
