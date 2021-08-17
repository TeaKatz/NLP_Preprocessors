import numpy as np

from typing import Union


class TokenIdPadding:
    padding_options = ["longest", "static_longest"]

    def __init__(self,
                padding_length: Union[str, int]="longest",
                padding_idx: int=0,
                return_padding_mask: bool=False,
                return_true_length: bool=False):

        if isinstance(padding_length, str):
            assert padding_length in self.padding_options

        self.padding_length = padding_length
        self.padding_idx = padding_idx
        self.return_padding_mask = return_padding_mask
        self.return_true_length = return_true_length

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
            true_lengths (optional):
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
        if self.return_true_length:
            true_lengths = np.zeros([batch_size], dtype=int)

        # Padding
        for i in range(batch_size):
            token_ids[i, :len(inputs[i])] = inputs[i][:padding_length]
            if self.return_padding_mask:
                padding_masks[i, :len(inputs[i])] = 0.
            if self.return_true_length:
                true_lengths[i] = min(len(inputs[i]), padding_length)

        # Return
        returns["token_ids"] = token_ids
        if self.return_padding_mask:
            returns["padding_masks"] = padding_masks
        if self.return_true_length:
            returns["true_lengths"] = true_lengths
        return returns


class WordTokenizerPadding(TokenIdPadding):
    pass

                
class CharacterLevelWordTokenizerPadding(TokenIdPadding):
    def __init__(self,
                padding_length: Union[str, int]="longest",
                char_padding_length: Union[str, int]="longest",
                padding_idx: int=0,
                return_padding_mask: bool=False,
                return_char_padding_mask: bool=False,
                return_true_length: bool=False,
                return_true_char_length: bool=False):

        if isinstance(char_padding_length, str):
            assert char_padding_length in self.padding_options

        super().__init__(padding_length, padding_idx, return_padding_mask, return_true_length)
        self.char_padding_length = char_padding_length
        self.return_char_padding_mask = return_char_padding_mask
        self.return_true_char_length = return_true_char_length

    def _get_char_padding_length(self, inputs):
        if self.char_padding_length == "static_longest":
            self.char_padding_length = char_padding_length = max([max([len(word) for word in words]) for words in inputs])
        elif self.char_padding_length == "longest":
            char_padding_length = max([max([len(word) for word in words]) for words in inputs])
        else:
            char_padding_length = self.char_padding_length
        return char_padding_length

    def __call__(self, inputs: list[list[list[int]]]):
        """
        inputs: (batch_size, *words_num, *word_length)
        *Dimension to pad

        Return dictionary of:
        {
            token_ids:
            padding_masks (optional):
            char_padding_masks (optional):
            true_lengths (optional):
            true_char_lengths (optional):
        }
        """
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Get char_padding_length
        char_padding_length = self._get_char_padding_length(inputs)

        # Initial
        returns = {}
        token_ids = np.full([batch_size, padding_length, char_padding_length], self.padding_idx, dtype=int)
        if self.return_padding_mask:
            padding_masks = np.ones([batch_size, padding_length], dtype=float)
        if self.return_char_padding_mask:
            char_padding_masks = np.ones([batch_size, padding_length, char_padding_length], dtype=float)
        if self.return_true_length:
            true_lengths = np.zeros([batch_size], dtype=int)
        if self.return_true_char_length:
            true_char_lengths = np.zeros([batch_size, padding_length], dtype=int)

        # Padding
        for i in range(batch_size):
            if self.return_padding_mask:
                padding_masks[i, :len(input[i])] = 0.
            if self.return_true_length:
                true_lengths[i] = min(len(inputs[i]), padding_length)

            for j in range(min(len(inputs[i]), padding_length)):
                token_ids[i, j, :len(inputs[i][j])] = inputs[i][j][:char_padding_length]
                if self.return_char_padding_mask:
                    char_padding_masks[i, j, :len(inputs[i][j])] = 0.
                if self.return_true_char_length:
                    true_char_lengths[i, j] = min(len(inputs[i][j]), char_padding_length)

        # Return
        returns["token_ids"] = token_ids
        if self.return_padding_mask:
            returns["padding_masks"] = padding_masks
        if self.return_char_padding_mask:
            returns["char_padding_masks"] = char_padding_masks
        if self.return_true_length:
            returns["true_lengths"] = true_lengths
        if self.return_true_char_length:
            returns["true_char_lengths"] = true_char_lengths
        return returns


class PositionalCharacterLevelWordTokenizerPadding(TokenIdPadding):
    def __init__(self,
                padding_length: Union[str, int]="longest",
                char_padding_length: Union[str, int]="longest",
                padding_idx: int=0,
                return_padding_mask: bool=False,
                return_char_padding_mask: bool=False,
                return_true_length: bool=False,
                return_true_char_length: bool=False):

        if isinstance(char_padding_length, str):
            assert char_padding_length in self.padding_options

        super().__init__(padding_length, padding_idx, return_padding_mask, return_true_length)
        self.char_padding_length = char_padding_length
        self.return_char_padding_mask = return_char_padding_mask
        self.return_true_char_length = return_true_char_length

    def _get_char_padding_length(self, inputs):
        if self.char_padding_length == "static_longest":
            self.char_padding_length = char_padding_length = max([max([len(word[0]) for word in words]) for words in inputs])
        elif self.char_padding_length == "longest":
            char_padding_length = max([max([len(word[0]) for word in words]) for words in inputs])
        else:
            char_padding_length = self.char_padding_length
        return char_padding_length

    def __call__(self, inputs: list[list[list[int], list[int]]]):
        """
        inputs: (batch_size, *words_num, 2, *word_length)
        *Dimension to pad

        Return dictionary of:
        {
            token_ids:
            position_ids:
            padding_masks (optional):
            char_padding_masks (optional):
            true_lengths (optional):
            true_char_lengths (optional):
        }
        """
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Get char_padding_length
        char_padding_length = self._get_char_padding_length(inputs)

        # Initial
        returns = {}
        token_ids = np.full([batch_size, padding_length, char_padding_length], self.padding_idx, dtype=int)
        position_ids = np.full([batch_size, padding_length, char_padding_length], 0, dtype=int)
        if self.return_padding_mask:
            padding_masks = np.ones([batch_size, padding_length], dtype=float)
        if self.return_char_padding_mask:
            char_padding_masks = np.ones([batch_size, padding_length, char_padding_length], dtype=float)
        if self.return_true_length:
            true_lengths = np.zeros([batch_size], dtype=int)
        if self.return_true_char_length:
            true_char_lengths = np.zeros([batch_size, padding_length], dtype=int)

        # Padding
        for i in range(batch_size):
            if self.return_padding_mask:
                padding_masks[i, :len(input[i])] = 0.
            if self.return_true_length:
                true_lengths[i] = min(len(inputs[i]), padding_length)

            for j in range(min(len(inputs[i]), padding_length)):
                token_ids[i, j, :len(inputs[i][j][0])] = inputs[i][j][0][:char_padding_length]
                position_ids[i, j, :len(inputs[i][j][1])] = inputs[i][j][1][:char_padding_length]
                if self.return_char_padding_mask:
                    char_padding_masks[i, j, :len(inputs[i][j][1])] = 0.
                if self.return_true_char_length:
                    true_char_lengths[i, j] = min(len(inputs[i][j][0]), char_padding_length)

        # Return
        returns["token_ids"] = token_ids
        returns["position_ids"] = position_ids
        if self.return_padding_mask:
            returns["padding_masks"] = padding_masks
        if self.return_char_padding_mask:
            returns["char_padding_masks"] = char_padding_masks
        if self.return_true_length:
            returns["true_lengths"] = true_lengths
        if self.return_true_char_length:
            returns["true_char_lengths"] = true_char_lengths
        return returns


class NgramLevelWordTokenizerPadding(CharacterLevelWordTokenizerPadding):
    pass