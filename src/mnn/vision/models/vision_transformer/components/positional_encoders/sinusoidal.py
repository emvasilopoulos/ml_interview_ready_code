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


def get_positional_encoding_tensor(number_of_tokens: int, size_of_token_embedding: int) -> torch.Tensor:
    positions_table = torch.tensor(
        [
            _get_position_angle_vec(i, size_of_token_embedding)
            for i in range(number_of_tokens)
        ]
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
        is_input_normalized: bool
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

        pos_enc_table = self.__initialize_positional_encoding()
        """
        Register the constant tensor as a buffer so that it is part of the module's state,
        but it is not a learnable parameter.
        """
        self.register_buffer('pe_buffer', pos_enc_table)  # Register positional encoding as buffer

        if is_input_normalized:
            """
            if input is normalized, i.e. in the range [0, 1]
            and custom positional encoding is in the range [0, 1],
            since we add both, the output will be in the range [0, 2].
            Thus we scale the output by 2 to keep it in the range [0, 1]   
            """
            self.scaling_factor = 2
        else:
            """
            if input is NOT normalized, i.e. in the range [0, 255]
            and custom positional encoding is in the range [0, 255],
            since we add both, the output will be in the range [0, 2 * 255] or [0, 510].
            Thus we scale the output by 2 * 255 to keep it in the range [0, 1]   
            """
            self.scaling_factor = 255 * 2

    def __initialize_positional_encoding(self) -> torch.Tensor:
        return self.__transform_positional_encoding_tensor(
            get_positional_encoding_tensor(
                self.number_of_tokens, self.size_of_token_embedding
            ),
            self.scaler,
        ).unsqueeze(0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        If two tensors x, y are “broadcastable”, the resulting tensor size is calculated as follows:
        If the number of dimensions of x and y are not equal,
        prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
        Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.
        Source: https://pytorch.org/docs/stable/notes/broadcasting.html

        In this example "pe" has shape (1, number_of_tokens, size_of_token_embedding) and "x" has shape (batch_size, number_of_tokens, size_of_token_embedding)
        The "pe" tensor is broadcasted to the shape (batch_size, number_of_tokens, size_of_token_embedding)

        Args:
            x: Input tensor of shape (batch_size, seq_len, size_of_token_embedding)
            or in other words (batch_size, height, width) # one channel
        Returns:
            x: The input tensor plus the positional encodings (same shape as input) normalized to [0, 1]
        """
        # Add positional encodings to the input tensor
        return (x + self.pe_buffer) / self.scaling_factor

def positional_encoding_tensor_to_opencv_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to an OpenCV image

    Args:
        tensor (torch.Tensor): tensor of shape (height, width, channels)

    Returns:
        np.ndarray: OpenCV image
    """
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# Test the positional encoding
def __preview_positional_encoding():
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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    myPe = MyVisionPositionalEncoding(10, 10, is_input_normalized=False)
    myPe.to(dtype=torch.float16)
    test_tensor = torch.rand(1, 10, 10, dtype=torch.float16) * 255
    print('output', myPe(test_tensor).dtype)

