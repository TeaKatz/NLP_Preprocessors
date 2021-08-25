import math
import numpy as np

from .utilities import array2str
from .BaseTokenizer import BaseTokenizer


class VectorTokenizer(BaseTokenizer):
    def __init__(self, 
                 num_embeddings: int,
                 vector_size: int,
                 padding_idx: int=0,
                 random_seed: int=0):

        super().__init__(num_embeddings, padding_idx)
        self.vector_size = vector_size

        np.random.seed(random_seed)
        self.random_vecs = np.random.normal(size=[math.ceil(math.log(num_embeddings, 2)), vector_size])

    def __call__(self, vectors: np.ndarray):
        """
        vectors: (batch_size, vector_size)
        return: (batch_size, )
        """
        return self.numerize(vectors)

    def numerize(self, vectors):
        binary_vecs = (vectors @ self.random_vecs.T > 0).astype(int)
        numbers = [int(array2str(vector), 2) % self.num_embeddings for vector in binary_vecs]
        numbers = [max(number, self.padding_idx + 1) for number in numbers]
        return  np.array(numbers)

    def tokenize(self):
        pass