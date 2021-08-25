import math
import numpy as np

from .utilities import array2str
from .BaseTokenizer import BaseTokenizer


class ImageTokenizer(BaseTokenizer):
    def __init__(self,
                 num_embeddings: int,
                 padding_idx: int=0,
                 window_height: int=9,
                 window_width: int=9,
                 stride: int=1,
                 padding_value: float=0,
                 random_seed: int=0):

        super().__init__(num_embeddings, padding_idx)
        self.window_height = window_height
        self.window_width = window_width
        self.stride = stride
        self.padding_value = padding_value

        np.random.seed(random_seed)
        self.random_vecs = np.random.normal(size=[math.ceil(math.log(num_embeddings, 2)), window_height * window_width])

    def __call__(self, images: list[np.ndarray]):
        return [self.numerize(self.tokenize(image)) for image in images]

    def tokenize(self, image: np.ndarray):
        """
        image: (height, width)
        return: (output_height, output_width, window_height, window_width)
        """
        height, width = image.shape

        # Calculate padding size
        output_height = math.ceil((height - self.window_height) / self.stride + 1)
        height_padding_size = (output_height - 1) * self.stride - height + self.window_height

        output_width = math.ceil((width - self.window_width) / self.stride + 1)
        width_padding_size = (output_width - 1) * self.stride - width + self.window_width

        # Padding
        image = np.pad(image, ((0, height_padding_size), (0, width_padding_size)), "constant", constant_values=self.padding_value)

        # Tokenize
        tokens = np.empty([output_height, output_width, self.window_height, self.window_width])
        for i in range(output_height):
            for j in range(output_width):
                # Get start and end indices
                start_y = i * self.stride
                end_y = start_y + self.window_height
                start_x = j * self.stride
                end_x = start_x + self.window_width
                # Get token
                tokens[output_height - i - 1, j] = image[start_y:end_y, start_x:end_x]
        return tokens

    def numerize(self, tokens: np.ndarray):
        """
        tokens: (output_height, output_width, window_height, window_width)
        return: (output_height, output_width)
        """
        # Reshape tokens
        output_height, output_width, _, _ = tokens.shape
        tokens = tokens.reshape(output_height, output_width, -1)

        # (output_height, output_width, log(num_embeddings, 2))
        binary_vecs = (tokens @ self.random_vecs.T > 0).astype(int)
        numbers = np.apply_along_axis(lambda x: int(array2str(x), 2) % self.num_embeddings, -1, binary_vecs)
        numbers = np.maximum(numbers, self.padding_idx + 1)
        return numbers