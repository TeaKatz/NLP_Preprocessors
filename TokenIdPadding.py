import numpy as np

from typing import Union


class TokenIdPadding:
    padding_options = ["longest", "static_longest"]

    def __init__(self,
                padding_length: Union[str, int],
                padding_idx: int=0,
                return_padding_mask: bool=False):

        if isinstance(padding_length, str):
            assert padding_length in self.padding_options

        self.padding_length = padding_length
        self.padding_idx = padding_idx
        self.return_padding_mask = return_padding_mask

    def __call__(self, inputs: list[list[int]]):
        batch_size = len(inputs)

        # Get padding_length
        if self.padding_length == "static_longest":
            self.padding_length = padding_length = max([len(ids) for ids in inputs])
        elif self.padding_length == "longest":
            padding_length = max([len(ids) for ids in inputs])
        else:
            padding_length = self.padding_length

        # Padding
        outputs = np.full([batch_size, padding_length], self.padding_idx)
        for i in range(batch_size):
            outputs[i, :len(inputs[i])] = inputs[i]
        return outputs
                
                
class NestedTokenIdPadding(TokenIdPadding):
    def __call__(self, inputs: list[list[list[int]]]):
        batch_size = len(inputs)

        # Get padding_length
        if self.padding_length == "static_longest":
            self.padding_length = padding_length = max([len(ids) for ids in inputs])
        elif self.padding_length == "longest":
            padding_length = max([len(ids) for ids in inputs])
        else:
            padding_length = self.padding_length

        # Get sub_padding_length
        sub_padding_length = max([max([len(sub_ids) for sub_ids in ids]) for ids in inputs])

        # Padding
        outputs = np.full([batch_size, padding_length, sub_padding_length], self.padding_idx)
        for i in range(batch_size):
            for j in range(len(inputs[i])):
                outputs[i, j, :len(inputs[i][j])] = inputs[i][j]
        return outputs
