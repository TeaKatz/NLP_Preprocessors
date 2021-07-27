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
        """
        inputs: (batch_size, *words_num)
        *Dimension to pad

        Return dictionary of:
        {
            token_ids:
            padding_masks (optional):
        }
        """
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Initial
        returns = {}
        token_ids = np.full([batch_size, padding_length], self.padding_idx, dtype=int)
        if self.return_padding_mask:
            padding_masks = np.ones([batch_size, padding_length], dtype=float)

        # Padding
        for i in range(batch_size):
            token_ids[i, :len(inputs[i])] = inputs[i][:padding_length]
            if self.return_padding_mask:
                padding_masks[i, :len(inputs[i])] = 0.

        # Return
        returns["token_ids"] = token_ids
        if self.return_padding_mask:
            returns["padding_masks"] = padding_masks
        return returns

                
class CharacterLevelWordTokenizerPadding(TokenIdPadding):
    def __init__(self,
                padding_length: Union[str, int],
                sub_padding_length: Union[str, int],
                padding_idx: int=0,
                return_padding_mask: bool=False,
                return_sub_padding_mask: bool=False):

        if isinstance(sub_padding_length, str):
            assert sub_padding_length in self.padding_options

        super().__init__(padding_length, padding_idx, return_padding_mask)
        self.sub_padding_length = sub_padding_length
        self.return_sub_padding_mask = return_sub_padding_mask

    def _get_sub_padding_length(self, inputs):
        if self.sub_padding_length == "static_longest":
            self.sub_padding_length = sub_padding_length = max([max([len(sub_items) for sub_items in items]) for items in inputs])
        elif self.sub_padding_length == "longest":
            sub_padding_length = max([max([len(sub_items) for sub_items in items]) for items in inputs])
        else:
            sub_padding_length = self.sub_padding_length
        return sub_padding_length

    def __call__(self, inputs: list[list[list[int]]]):
        """
        inputs: (batch_size, *words_num, *word_length)
        *Dimension to pad

        Return dictionary of:
        {
            token_ids:
            padding_masks (optional):
        }
        """
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Get sub_padding_length
        sub_padding_length = self._get_sub_padding_length(inputs)

        # Initial
        returns = {}
        token_ids = np.full([batch_size, padding_length, sub_padding_length], self.padding_idx, dtype=int)
        if self.return_padding_mask:
            padding_masks = np.ones([batch_size, padding_length], dtype=float)
        if self.return_sub_padding_mask:
            sub_padding_masks = np.ones([batch_size, padding_length, sub_padding_length], dtype=float)

        # Padding
        for i in range(batch_size):
            for j in range(min(len(inputs[i]), padding_length)):
                token_ids[i, j, :len(inputs[i][j])] = inputs[i][j][:sub_padding_length]
                if self.return_padding_mask:
                    padding_masks[i, j] = 0.
                if self.return_sub_padding_mask:
                    sub_padding_masks[i, j, :len(inputs[i][j])] = 0.

        # Return
        returns["token_ids"] = token_ids
        if self.return_padding_mask:
            returns["padding_masks"] = padding_masks
        if self.return_sub_padding_mask:
            returns["sub_padding_masks"] = sub_padding_masks
        return returns


class PositionalCharacterLevelWordTokenizerPadding(TokenIdPadding):
    def __init__(self,
                padding_length: Union[str, int],
                sub_padding_length: Union[str, int],
                padding_idx: int=0,
                return_padding_mask: bool=False,
                return_sub_padding_mask: bool=False):

        if isinstance(sub_padding_length, str):
            assert sub_padding_length in self.padding_options

        super().__init__(padding_length, padding_idx, return_padding_mask)
        self.sub_padding_length = sub_padding_length
        self.return_sub_padding_mask = return_sub_padding_mask

    def _get_sub_padding_length(self, inputs):
        if self.sub_padding_length == "static_longest":
            self.sub_padding_length = sub_padding_length = max([max([len(sub_items[0]) for sub_items in items]) for items in inputs])
        elif self.sub_padding_length == "longest":
            sub_padding_length = max([max([len(sub_items[0]) for sub_items in items]) for items in inputs])
        else:
            sub_padding_length = self.sub_padding_length
        return sub_padding_length

    def __call__(self, inputs: list[list[list[int], list[int]]]):
        """
        inputs: (batch_size, *words_num, 2, *word_length)
        *Dimension to pad

        Return dictionary of:
        {
            token_ids:
            position_ids:
            padding_masks (optional):
        }
        """
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Get sub_padding_length
        sub_padding_length = self._get_sub_padding_length(inputs)

        # Initial
        returns = {}
        token_ids = np.full([batch_size, padding_length, sub_padding_length], self.padding_idx, dtype=int)
        position_ids = np.full([batch_size, padding_length, sub_padding_length], 0, dtype=int)
        if self.return_padding_mask:
            padding_masks = np.ones([batch_size, padding_length], dtype=float)
        if self.return_sub_padding_mask:
            sub_padding_masks = np.ones([batch_size, padding_length, sub_padding_length], dtype=float)

        # Padding
        for i in range(batch_size):
            for j in range(min(len(inputs[i]), padding_length)):
                token_ids[i, j, :len(inputs[i][j][0])] = inputs[i][j][0][:sub_padding_length]
                position_ids[i, j, :len(inputs[i][j][1])] = inputs[i][j][1][:sub_padding_length]
                if self.return_padding_mask:
                    padding_masks[i, j] = 0.
                if self.return_sub_padding_mask:
                    sub_padding_masks[i, j, :len(inputs[i][j][1])] = 0.

        # Return
        returns["token_ids"] = token_ids
        returns["position_ids"] = position_ids
        if self.return_padding_mask:
            returns["padding_masks"] = padding_masks
        if self.return_sub_padding_mask:
            returns["sub_padding_masks"] = sub_padding_masks
        return returns


class NgramLevelWordTokenizerPadding(CharacterLevelWordTokenizerPadding):
    pass