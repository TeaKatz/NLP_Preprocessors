import math
import librosa
import numpy as np

from .utilities import array2str
from .utilities import shorten_signal
from .BaseTokenizer import BaseTokenizer
from .ImageTokenizer import ImageTokenizer


class SignalTokenizer(BaseTokenizer):
    def __init__(self, 
                 num_embeddings: int,
                 padding_idx: int=0,
                 window_size: int=1000,
                 stride: int=100,
                 padding_value: float=0.0,
                 shorten_threshold: float=1e-3,
                 shorten_offset: int=500,
                 random_seed: int=0):

        super().__init__(num_embeddings, padding_idx)
        self.window_size = window_size
        self.stride = stride
        self.padding_value = padding_value
        self.shorten_threshold = shorten_threshold
        self.shorten_offset = shorten_offset

        np.random.seed(random_seed)
        self.random_vecs = np.random.normal(size=[math.ceil(math.log(num_embeddings, 2)), window_size])

    def __call__(self, signals: list[np.ndarray]):
        return [self.numerize(self.tokenize(signal)) for signal in signals]

    def tokenize(self, signal: np.ndarray):
        """
        signal: (signal_length, )
        return: (output_length, window_size)
        """
        signal = shorten_signal(signal, threshold=self.shorten_threshold, offset=self.shorten_offset)
        signal_length = signal.shape[0]

        # Calculate padding size
        output_length = math.ceil((signal_length - self.window_size) / self.stride + 1)
        padding_size = (output_length - 1) * self.stride - signal_length + self.window_size
        # Padding
        signal = np.pad(signal, (0, padding_size), "constant", constant_values=self.padding_value)
        # Tokenize
        tokens = np.concatenate([signal[np.newaxis, i * self.stride:i * self.stride + self.window_size] for i in range(output_length)], axis=0)
        return tokens

    def numerize(self, tokens: np.ndarray):
        """
        tokens: (output_length, window_size)
        return: (output_length, )
        """
        binary_vecs = (tokens @ self.random_vecs.T > 0).astype(int)
        numbers = [int(array2str(vector), 2) % self.num_embeddings for vector in binary_vecs]
        numbers = [max(number, self.padding_idx + 1) for number in numbers]
        return  np.array(numbers)


class SignalDerivativeTokenizer(SignalTokenizer):
    def tokenize(self, signal: np.ndarray):
        """
        signal: (signal_length, )
        return: (output_length, window_size)
        """
        signal = shorten_signal(signal, threshold=self.shorten_threshold, offset=self.shorten_offset)
        signal = signal[1:] - signal[:-1]
        signal_length = signal.shape[0]

        # Calculate padding size
        output_length = math.ceil((signal_length - self.window_size) / self.stride + 1)
        padding_size = (output_length - 1) * self.stride - signal_length + self.window_size
        # Padding
        signal = np.pad(signal, (0, padding_size), "constant", constant_values=self.padding_value)
        # Tokenize
        tokens = np.concatenate([signal[np.newaxis, i * self.stride:i * self.stride + self.window_size] for i in range(output_length)], axis=0)
        return tokens


class SignalSpectrogramTokenizer(ImageTokenizer):
    def __init__(self,
                 num_embeddings: int,
                 sampling_rate: int=22050,
                 n_fft: int=2000,
                 hop_length: int=100,
                 padding_idx: int=0,
                 window_height: int=9,
                 window_width: int=9,
                 stride: int=1,
                 padding_value: float=0,
                 shorten_threshold: float=1e-3,
                 shorten_offset: int=500,
                 random_seed: int=0):

        super().__init__(num_embeddings, padding_idx, window_height, window_width, stride, padding_value, random_seed)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.shorten_threshold = shorten_threshold
        self.shorten_offset = shorten_offset

    def tokenize(self, signal: np.ndarray):
        """
        signal: (signal_length, )
        return: (output_height, output_width, window_height, window_width)
        """
        # Convert signal into spectrogram image
        signal = shorten_signal(signal, threshold=self.shorten_threshold, offset=self.shorten_offset)
        spectrogram = librosa.feature.melspectrogram(y=signal, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram = (spectrogram + 40) / 40
        return super().tokenize(spectrogram)