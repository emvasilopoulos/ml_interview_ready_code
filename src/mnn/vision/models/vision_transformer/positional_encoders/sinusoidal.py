import cv2
import numpy as np
import torch
import torch.nn


# https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
def _get_position_angle_vec(i: int, token_len: int):
    return [i / 10000 ** (2 * j / token_len) for j in range(token_len)]


def get_sinusoid_table(num_tokens: int, token_len: int) -> np.ndarray:
    """Make Sinusoid Table

    Args:
        num_tokens (int): number of tokens
        token_len (int): length of a token

    Returns:
        (torch.FloatTensor) sinusoidal position encoding table
    """

    sinusoid_table = np.array(
        [_get_position_angle_vec(i, token_len) for i in range(num_tokens)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return sinusoid_table


def get_positional_encoding_tensor(number_of_tokens: int, size_of_token_embedding: int):
    positions_table = torch.tensor(
        [
            _get_position_angle_vec(i, size_of_token_embedding)
            for i in range(number_of_tokens)
        ],
        dtype=torch.float,
    )

    pe = torch.zeros(number_of_tokens, size_of_token_embedding)
    pe[:, 0::2] = torch.sin(positions_table[:, 0::2])
    pe[:, 1::2] = torch.cos(positions_table[:, 1::2])

    return pe


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

        self.pe = get_positional_encoding_tensor(
            number_of_tokens, size_of_token_embedding
        ).unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, size_of_token_embedding)
        Returns:
            x: The input tensor plus the positional encodings (same shape as input)
        """
        # Add positional encodings to the input tensor
        return x + self.pe


class MyVisionPositionalEncoding(torch.nn.Module):
    def __transform_positional_encoding_tensor(
        self, pe: torch.Tensor, scaler: int
    ) -> torch.Tensor:
        """
        Context: Initially the positional encoding tensor is in the range [-1, 1] due to the sin and cos functions
        Transform the positional encoding tensor to be in the range [0, 1]
        or in the case of the input not being normalized,
        transform the positional encoding tensor to be in the range [0, 255]

        Args:
            pe (torch.Tensor): tensor of shape (number_of_tokens, size_of_token_embedding)

        Returns:
            torch.Tensor: tensor of shape (number_of_tokens, size_of_token_embedding)
        """
        return ((pe + 1) * scaler) / 2

    def __init__(
        self,
        number_of_tokens: int,
        size_of_token_embedding: int,
        is_input_normalized: bool,
    ):
        """
        Args:
            d_model: Dimension of the model (the embedding size).
            max_len: Maximum length of the sequence.
        """
        super().__init__()

        if is_input_normalized:
            scaler = 1
        else:
            scaler = 255
        self.scaler = scaler
        self.number_of_tokens = number_of_tokens
        self.size_of_token_embedding = size_of_token_embedding

        self.pe = self.__initialize_positional_encoding()

        self.__is_input_normalized = is_input_normalized

    def __initialize_positional_encoding(self):
        return self.__transform_positional_encoding_tensor(
            get_positional_encoding_tensor(
                self.number_of_tokens, self.size_of_token_embedding
            ),
            self.scaler,
        ).unsqueeze(0)

    def set_batch_size(self, batch_size: int):
        self.pe = self.__initialize_positional_encoding()
        self.pe = self.pe.repeat(batch_size, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, size_of_token_embedding)
            or in other words (batch_size, height, width) # one channel
        Returns:
            x: The input tensor plus the positional encodings (same shape as input) normalized to [0, 1]
        """
        # Add positional encodings to the input tensor
        if self.__is_input_normalized:
            return (x + self.pe) / 2
        else:
            return (x + self.pe) / (255 * 2)


def positional_encoding_tensor_to_opencv_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to an OpenCV image

    Args:
        tensor (torch.Tensor): tensor of shape (height, width, channels)

    Returns:
        np.ndarray: OpenCV image
    """
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


if __name__ == "__main__":
    # Test the positional encoding
    num_tokens = 640
    token_lens = [640, 768, 896, 1024]
    images = []
    for i, token_len in enumerate(token_lens):
        my_pos_enc = MyVisionPositionalEncoding(
            num_tokens, token_len, is_input_normalized=False
        )

        image = torch.rand(1, num_tokens, token_len) * 255
        output = my_pos_enc(image)

        out_image = positional_encoding_tensor_to_opencv_image(output)
        out_image = cv2.rotate(
            cv2.cvtColor(out_image[:, :, 0], cv2.COLOR_GRAY2BGR),
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )
        cv2.imshow(f"output-{token_len}", out_image)
    cv2.waitKey(0)
