import torch
import numpy as np

from logging import raiseExceptions
from typing import Union, List


class TokenIdPadding:
    padding_options = ["longest", "static_longest"]
    tensor_options = ["pt"]

    def __init__(self,
                padding_length: Union[str, int]="longest",
                max_padding_length: int=None,
                padding_id: int=0,
                **kwargs):

        if isinstance(padding_length, str):
            assert padding_length in self.padding_options

        self.padding_length = padding_length
        self.padding_id = padding_id
        self.max_padding_length = max_padding_length

    def _get_padding_length(self, inputs):
        if self.padding_length == "static_longest":
            self.padding_length = padding_length = max([len(items) for items in inputs])
        elif self.padding_length == "longest":
            padding_length = max([len(items) for items in inputs])
        else:
            padding_length = self.padding_length

        if self.max_padding_length is not None:
            padding_length = min(padding_length, self.max_padding_length)
        return padding_length

    def __call__(self, 
                inputs: List[List[int]], 
                return_padding_masks: bool=False, 
                return_true_lengths: bool=False, 
                return_dict: bool=False, 
                return_tensors: Union[str, bool]=False,
                **kwargs):
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
        token_ids = np.full([batch_size, padding_length], self.padding_id, dtype=int)
        if return_padding_masks:
            padding_masks = np.zeros([batch_size, padding_length], dtype=int)
        if return_true_lengths:
            true_lengths = np.zeros([batch_size], dtype=int)

        # Padding
        for i in range(batch_size):
            token_ids[i, :len(inputs[i])] = inputs[i][:padding_length]
            if return_padding_masks:
                padding_masks[i, :len(inputs[i])] = 1
            if return_true_lengths:
                true_lengths[i] = min(len(inputs[i]), padding_length)

        # Tensor
        if return_tensors == "pt":
            token_ids = torch.tensor(token_ids).long()
            if return_padding_masks:
                padding_masks = torch.tensor(padding_masks).long()
            if return_true_lengths:
                true_lengths = torch.tensor(true_lengths).long()

        # Return
        if return_dict:
            returns["token_ids"] = token_ids
            if return_padding_masks:
                returns["padding_masks"] = padding_masks
            if return_true_lengths:
                returns["true_lengths"] = true_lengths
            return returns
        else:
            returns = (token_ids, )
            if return_padding_masks:
                returns += (padding_masks, )
            if return_true_lengths:
                returns += (true_lengths, )
            return returns[0] if len(returns) == 1 else returns


class SubTokenIdPadding(TokenIdPadding):
    def __init__(self,
                padding_length: Union[str, int]="longest",
                sub_padding_length: Union[str, int]="longest",
                max_padding_length: int=None,
                max_sub_padding_length: int=None,
                padding_id: int=0,
                **kwargs):

        super().__init__(padding_length=padding_length,
                         max_padding_length=max_padding_length,
                         padding_id=padding_id)

        if isinstance(sub_padding_length, str):
            assert sub_padding_length in self.padding_options

        self.sub_padding_length = sub_padding_length
        self.max_sub_padding_length=max_sub_padding_length

    def _get_sub_padding_length(self, inputs):
        if self.sub_padding_length == "static_longest":
            self.sub_padding_length = sub_padding_length = max([max([len(item) for item in items]) for items in inputs])
        elif self.sub_padding_length == "longest":
            sub_padding_length = max([max([len(item) for item in items]) for items in inputs])
        else:
            sub_padding_length = self.sub_padding_length

        if self.max_sub_padding_length is not None:
            sub_padding_length = min(sub_padding_length, self.max_sub_padding_length)
        return sub_padding_length

    def __call__(self, 
                inputs: List[List[List[int]]],
                return_padding_masks: bool=False, 
                return_sub_padding_masks: bool=False,
                return_true_lengths: bool=False, 
                return_true_sub_lengths: bool=False,
                return_dict: bool=False, 
                return_tensors: Union[str, bool]=False,
                **kwargs):
        """
        inputs: (batch_size, *words_num, *subwords_num)
        *Dimension to pad

        Return dictionary of:
        {
            token_ids: (batch_size, words_num, subwords_num)
            padding_masks (optional): (batch_size, words_num)
            sub_padding_masks (optional): (batch_size, words_num, subwords_num)
            true_lengths (optional): (batch_size, )
            true_sub_lengths (optional): (batch_size, words_num)
        }
        """
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Get sub_padding_length
        sub_padding_length = self._get_sub_padding_length(inputs)

        # Initial
        returns = {}
        token_ids = np.full([batch_size, padding_length, sub_padding_length], self.padding_id, dtype=int)
        if return_padding_masks:
            padding_masks = np.zeros([batch_size, padding_length], dtype=int)
        if return_sub_padding_masks:
            sub_padding_masks = np.zeros([batch_size, padding_length, sub_padding_length], dtype=int)
        if return_true_lengths:
            true_lengths = np.zeros([batch_size], dtype=int)
        if return_true_sub_lengths:
            true_sub_lengths = np.zeros([batch_size, padding_length], dtype=int)

        # Padding
        for i in range(batch_size):
            if return_padding_masks:
                padding_masks[i, :len(inputs[i])] = 1
            if return_true_lengths:
                true_lengths[i] = min(len(inputs[i]), padding_length)

            for j in range(min(len(inputs[i]), padding_length)):
                token_ids[i, j, :len(inputs[i][j])] = inputs[i][j][:sub_padding_length]
                if return_sub_padding_masks:
                    sub_padding_masks[i, j, :len(inputs[i][j])] = 1
                if return_true_sub_lengths:
                    true_sub_lengths[i, j] = min(len(inputs[i][j]), sub_padding_length)

        # Tensor
        if return_tensors == "pt":
            token_ids = torch.tensor(token_ids).long()
            if return_padding_masks:
                padding_masks = torch.tensor(padding_masks).long()
            if return_sub_padding_masks:
                sub_padding_masks = torch.tensor(sub_padding_masks).long()
            if return_true_lengths:
                true_lengths = torch.tensor(true_lengths).long()
            if return_true_sub_lengths:
                true_sub_lengths = torch.tensor(true_sub_lengths).long()

        # Return
        if return_dict:
            returns["token_ids"] = token_ids
            if return_padding_masks:
                returns["padding_masks"] = padding_masks
            if return_sub_padding_masks:
                returns["sub_padding_masks"] = sub_padding_masks
            if return_true_lengths:
                returns["true_lengths"] = true_lengths
            if return_true_sub_lengths:
                returns["true_sub_lengths"] = true_sub_lengths
            return returns
        else:
            returns = (token_ids, )
            if return_padding_masks:
                returns += (padding_masks, )
            if return_sub_padding_masks:
                returns += (sub_padding_masks, )
            if return_true_lengths:
                returns += (true_lengths, )
            if return_true_sub_lengths:
                returns += (true_sub_lengths, )
            return returns[0] if len(returns) == 1 else returns


class AutoTokenIdPadding:
    def __init__(self, *args, **kwargs):
        self.token_id_padding = TokenIdPadding(*args, **kwargs)
        self.subtoken_id_padding = SubTokenIdPadding(*args, **kwargs)

    def __call__(self, inputs: List, **kwargs):
        if isinstance(inputs[0], int):
            # Inputs cannot be padded
            return inputs   
        elif isinstance(inputs[0][0], int):
            return self.token_id_padding(inputs, **kwargs)
        elif isinstance(inputs[0][0][0], int):
            return self.subtoken_id_padding(inputs, **kwargs)
        else:
            raiseExceptions("inputs format doesn't match")


class WordTokenizerPadding(TokenIdPadding):
    pass

                
class CharacterLevelWordTokenizerPadding(SubTokenIdPadding):
    pass


class PositionalCharacterLevelWordTokenizerPadding(TokenIdPadding):
    def __init__(self,
                padding_length: Union[str, int]="longest",
                char_padding_length: Union[str, int]="longest",
                max_padding_length: int=None,
                max_char_padding_length: int=None,
                padding_idx: int=0,
                return_padding_mask: bool=False,
                return_char_padding_mask: bool=False,
                return_true_length: bool=False,
                return_true_char_length: bool=False):

        if isinstance(char_padding_length, str):
            assert char_padding_length in self.padding_options

        super().__init__(padding_length, max_padding_length, padding_idx, return_padding_mask, return_true_length)
        self.char_padding_length = char_padding_length
        self.max_char_padding_length = max_char_padding_length
        self.return_char_padding_mask = return_char_padding_mask
        self.return_true_char_length = return_true_char_length

    def _get_char_padding_length(self, inputs):
        if self.char_padding_length == "static_longest":
            self.char_padding_length = char_padding_length = max([max([len(word[0]) for word in words]) for words in inputs])
        elif self.char_padding_length == "longest":
            char_padding_length = max([max([len(word[0]) for word in words]) for words in inputs])
        else:
            char_padding_length = self.char_padding_length

        if self.max_char_padding_length is not None:
            char_padding_length = min(char_padding_length, self.max_char_padding_length)
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
                padding_masks[i, :len(inputs[i])] = 0.
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