import copy
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

    def _get_padding_length(self, inputs):
        if self.padding_length == "static_longest":
            self.padding_length = padding_length = max([len(items) for items in inputs])
        elif self.padding_length == "longest":
            padding_length = max([len(items) for items in inputs])
        else:
            padding_length = self.padding_length
        return padding_length

    def __call__(self, inputs: list[list[int]]):
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Initial returns
        returns = {}

        # Padding
        token_ids = np.full([batch_size, padding_length], self.padding_idx)
        for i in range(batch_size):
            token_ids[i, :len(inputs[i])] = inputs[i]
        returns["token_ids"] = token_ids

        # Get padding_mask
        if self.return_padding_mask:
            padding_masks = (token_ids == 0).astype(float)
            returns["padding_masks"] = padding_masks
        return returns
                
                
class CharacterLevelWordTokenizerPadding(TokenIdPadding):
    def __call__(self, inputs: list[list[list[int]]]):
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Get sub_padding_length
        sub_padding_length = max([max([len(sub_items) for sub_items in items]) for items in inputs])

        # Initial returns
        returns = {}

        # Padding
        token_ids = np.full([batch_size, padding_length, sub_padding_length], self.padding_idx)
        for i in range(batch_size):
            for j in range(len(inputs[i])):
                token_ids[i, j, :len(inputs[i][j])] = inputs[i][j]
        returns["token_ids"] = token_ids

        # Get padding_mask
        if self.return_padding_mask:
            padding_masks = (token_ids == 0).astype(float)
            returns["padding_masks"] = padding_masks
        return returns


class PositionalCharacterLevelWordTokenizerPadding(TokenIdPadding):
    def __call__(self, inputs: list[list[list[list[int], list[int]]]]):
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Get sub_padding_length
        sub_padding_length = max([max([len(sub_items[0]) for sub_items in items]) for items in inputs])

        # Initial returns
        returns = {}

        # Padding
        token_ids = np.full([batch_size, padding_length, sub_padding_length], self.padding_idx)
        position_ids = np.full([batch_size, padding_length, sub_padding_length], 0)
        for i in range(batch_size):
            for j in range(len(inputs[i])):
                token_ids[i, j, :len(inputs[i][j][0])] = inputs[i][j][0]
                position_ids[i, j, :len(inputs[i][j][1])] = inputs[i][j][1]
        returns["token_ids"] = token_ids
        returns["position_ids"] = position_ids

        # Get padding_mask
        if self.return_padding_mask:
            padding_masks = (token_ids == 0).astype(float)
            returns["padding_masks"] = padding_masks
        return returns

