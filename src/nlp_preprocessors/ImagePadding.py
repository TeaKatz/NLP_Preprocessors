import numpy as np

from typing import Union


class ImagePadding:
    padding_options = ["longest", "static_longest"]

    def __init__(self,
                padding_height: Union[str, int]="longest",
                padding_width: Union[str, int]="longest",
                return_padding_mask: bool=False,
                return_true_length: bool=False):

        if isinstance(padding_height, str):
            assert padding_height in self.padding_options
        if isinstance(padding_width, str):
            assert padding_width in self.padding_options

        self.padding_height = padding_height
        self.padding_width = padding_width
        self.return_padding_mask = return_padding_mask
        self.return_true_length = return_true_length

    def _get_padding_height(self, inputs):
        if self.padding_height == "static_longest":
            self.padding_height = padding_height = max(image.shape[0] for image in inputs)
        elif self.padding_height == "longest":
            padding_height = max(image.shape[0] for image in inputs)
        else:
            padding_height = self.padding_height
        return padding_height

    def _get_padding_width(self, inputs):
        if self.padding_width == "static_longest":
            self.padding_width = padding_width = max(image.shape[1] for image in inputs)
        elif self.padding_width == "longest":
            padding_width = max(image.shape[1] for image in inputs)
        else:
            padding_width = self.padding_width
        return padding_width

    def __call__(self, inputs: np.ndarray):
        """
        inputs: (batch_size, *height, *width)
        *Dimension to pad

        Return dictionary of:
        {
            images:
            padding_masks (optional):
            true_lengths (optional):
        }
        """
        batch_size = len(inputs)

        # Get padding_height
        padding_height = self._get_padding_height(inputs)
        # Get padding_width
        padding_width = self._get_padding_width(inputs)

        # Initial
        returns = {}
        images = np.zeros([batch_size, padding_height, padding_width], dtype=float)
        if self.return_padding_mask:
            padding_masks = np.ones([batch_size, padding_height, padding_width], dtype=float)
        if self.return_true_length:
            true_lengths = np.zeros([batch_size], dtype=int)

        # Padding
        for i in range(batch_size):
            images[i, :inputs[i].shape[0], :inputs[i].shape[1]] = inputs[i, :padding_height, :padding_width]
            if self.return_padding_mask:
                padding_masks[i, :inputs[i].shape[0], :inputs[i].shape[1]] = 0.
            if self.return_true_length:
                true_lengths[i] = (min(inputs[i].shape[0], padding_height), min(inputs[i].shape[1], padding_width))

        # Return
        returns["images"] = images
        if self.return_padding_mask:
            returns["padding_masks"] = padding_masks
        if self.return_true_length:
            returns["true_lengths"] = true_lengths
        return returns
