import numpy as np

from typing import Union


class SignalPadding:
    padding_options = ["longest", "static_longest"]

    def __init__(self,
                padding_length: Union[str, int]="longest",
                return_padding_mask: bool=False,
                return_true_length: bool=False):

        if isinstance(padding_length, str):
            assert padding_length in self.padding_options

        self.padding_length = padding_length
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
        inputs: (batch_size, *signal_length)
        *Dimension to pad

        Return dictionary of:
        {
            signals:
            padding_masks (optional):
            true_lengths (optional):
        }
        """
        batch_size = len(inputs)

        # Get padding_length
        padding_length = self._get_padding_length(inputs)

        # Initial
        returns = {}
        signals = np.zeros([batch_size, padding_length], dtype=float)
        if self.return_padding_mask:
            padding_masks = np.ones([batch_size, padding_length], dtype=float)
        if self.return_true_length:
            true_lengths = np.zeros([batch_size], dtype=int)

        # Padding
        for i in range(batch_size):
            signals[i, :len(inputs[i])] = inputs[i][:padding_length]
            if self.return_padding_mask:
                padding_masks[i, :len(inputs[i])] = 0.
            if self.return_true_length:
                true_lengths[i] = min(len(inputs[i]), padding_length)

        # Return
        returns["signals"] = signals
        if self.return_padding_mask:
            returns["padding_masks"] = padding_masks
        if self.return_true_length:
            returns["true_lengths"] = true_lengths
        return returns
