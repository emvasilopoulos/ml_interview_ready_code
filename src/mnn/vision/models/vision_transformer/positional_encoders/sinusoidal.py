import numpy as np
import torch
import torch.nn


# https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
def _get_position_angle_vec(i):
    return [i / 10000 ** (2 * j / token_len) for j in range(token_len)]


def get_sinusoid_table(num_tokens: int, token_len: int) -> np.ndarray:
    """Make Sinusoid Table

    Args:
        num_tokens (int): number of tokens
        token_len (int): length of a token

    Returns:
        (torch.FloatTensor) sinusoidal position encoding table
    """

    sinusoid_table = np.array([_get_position_angle_vec(i) for i in range(num_tokens)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table


class PositionalEncoding(torch.nn.Module):
    """
    Functionally, the position embedding is a matrix with the same shape as the tokens
    Tokens have a shape --> (number_of_tokens, size_of_token_embedding)
    """

    def __init__(self, number_of_tokens: int, size_of_token_embedding: int):
        super().__init__()
        # The paper suggests using 1D positional embeddings
        # 2D positional embeddings do not have any advantages
        self.layer = torch.tensor(
            get_sinusoid_table(number_of_tokens, size_of_token_embedding),
            dtype=torch.float,
        ).unsqueeze(0)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return patches + self.layer


class PositionalEncoding2(torch.nn.Module):
    def __init__(self, number_of_tokens: int, size_of_token_embedding: int):
        """
        Args:
            d_model: Dimension of the model (the embedding size).
            max_len: Maximum length of the sequence.
        """
        super().__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(number_of_tokens, size_of_token_embedding)

        sinusoid_table = torch.tensor(
            [_get_position_angle_vec(i) for i in range(num_tokens)], dtype=torch.float
        )
        pe[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        pe[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, size_of_token_embedding)
        Returns:
            x: The input tensor plus the positional encodings (same shape as input)
        """
        # Add positional encodings to the input tensor
        return x + self.pe


if __name__ == "__main__":
    # Test the positional encoding
    num_tokens = 100
    token_len = 8

    pos_enc = PositionalEncoding(num_tokens, token_len)
    pos_enc2 = PositionalEncoding2(num_tokens, token_len)

    assert torch.isclose(pos_enc.layer[0, 1, :], pos_enc2.pe[0, 1, :]).all()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    plt.plot(np.arange(num_tokens), pos_enc.layer[0, :, :].numpy())
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.title("Positional Encoding")
    plt.legend([f"dim {i}" for i in range(pos_enc.layer.shape[-1])])
    plt.show()
    plt.close()
1
