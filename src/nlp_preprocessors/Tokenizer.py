import os
import math
import joblib
import hashlib
import librosa

import numpy as np

from tqdm import tqdm
from abc import abstractmethod
from nltk import word_tokenize

from .utilities import Word2Syllable
from .utilities import word2ngram
from .utilities import word2skipngram
from .utilities import LocalitySensitiveHashing
from .utilities import shorten_signal
from .utilities import array2str


class BaseTokenizer:
    def __init__(self, 
                num_embeddings: int, 
                padding_idx: int=0):

        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx

    def __call__(self, inputs: list):
        return [[self.numerize(token) for token in self.tokenize(inp)] for inp in inputs]

    @abstractmethod
    def tokenize(self, inp):
        """ Convert a given input into a sequence of tokens """
        pass

    @abstractmethod
    def numerize(self, token):
        """ Convert a given token into a number """
        pass


class TextTokenizer(BaseTokenizer):
    special_tokens = ["<PAD>", "<CLS>", "<SEP>", "<MASK>", "<UNK>"]

    def __init__(self, 
                num_embeddings: int, 
                padding_idx: int=0,
                input_word: bool=False):

        super().__init__(num_embeddings, padding_idx)
        self.input_word = input_word


class HashingBasedTokenizer(TextTokenizer):
    def numerize(self, token: str):
        """ Convert a given token into a number """
        hash_number = int(hashlib.sha3_224(bytes(token, "utf8")).hexdigest(), 16) % self.num_embeddings
        hash_number = max(hash_number, self.padding_idx + len(self.special_tokens))
        return hash_number


class LocalitySensitiveHashingBasedTokenizer(TextTokenizer):
    def __init__(self, 
                 num_embeddings: int, 
                 padding_idx: int=0,
                 random_seed: int=0):

        super().__init__(num_embeddings, padding_idx)
        self.lsh = LocalitySensitiveHashing(num_embeddings, random_seed)

    def numerize(self, token: str):
        """ Convert a given token into a number """
        hash_number = self.lsh(token) % self.num_embeddings
        hash_number = max(hash_number, self.padding_idx + len(self.special_tokens))
        return hash_number


class NgramLevelWordTokenizer(HashingBasedTokenizer):
    def __init__(self, 
                 num_embeddings: int, 
                 padding_idx: int=0,
                 ngrams: list=(3, 4, 5, 6),
                 skipngrams: list=(2, 3)):
        
        super().__init__(num_embeddings, padding_idx)
        self.ngrams = ngrams
        self.skipngrams = skipngrams

    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        words = word_tokenize(string) if not self.input_word else [string]
        tokens = []
        for word in words:
            grams = []
            for n in self.ngrams:
                grams.extend(word2ngram(word, n))
            for n in self.skipngrams:
                grams.extend(word2skipngram(word, n))
            tokens.append(grams)
        return tokens

    def numerize(self, grams: list[str]):
        """ Convert a given list of tokens into a list of numbers """
        sub = super()
        return [sub.numerize(gram) for gram in grams]


class LshNgramLevelWordTokenizer(LocalitySensitiveHashingBasedTokenizer):
    def __init__(self, 
                 num_embeddings: int, 
                 padding_idx: int=0,
                 random_seed: int=0,
                 ngrams: list=(3, 4, 5, 6),
                 skipngrams: list=(2, 3)):
        
        super().__init__(num_embeddings, padding_idx, random_seed)
        self.ngrams = ngrams
        self.skipngrams = skipngrams

    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        words = word_tokenize(string) if not self.input_word else [string]
        tokens = []
        for word in words:
            grams = []
            for n in self.ngrams:
                grams.extend(word2ngram(word, n))
            for n in self.skipngrams:
                grams.extend(word2skipngram(word, n))
            tokens.append(grams)
        return tokens

    def numerize(self, grams: list[str]):
        """ Convert a given list of tokens into a list of numbers """
        sub = super()
        return [sub.numerize(gram) for gram in grams]


class CharacterLevelWordTokenizer(HashingBasedTokenizer):
    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        words = word_tokenize(string) if not self.input_word else [string]
        tokens = [list(word) for word in words]
        return tokens

    def numerize(self, chars: list[str]):
        """ Convert a given list of tokens into a list of numbers """
        sub = super()
        return [sub.numerize(char) for char in chars]


class PositionalCharacterLevelWordTokenizer(HashingBasedTokenizer):
    def __init__(self, 
                num_embeddings: int,
                padding_idx: int=0,
                max_positional: int=10):

        super().__init__(num_embeddings, padding_idx)
        self.max_positional = max_positional

    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        words = word_tokenize(string) if not self.input_word else [string]
        tokens = [self.positionize(list(word)) for word in words]
        return tokens

    def numerize(self, token: list[str]):
        """ Convert a given list of tokens into a list of tuples of number and position """
        sub = super()
        chars, positions = token
        numbers = [sub.numerize(char) for char in chars]
        return numbers, positions

    @abstractmethod
    def positionize(self, chars: list[str]):
        pass


class RoughPositionalCharacterLevelWordTokenizer(PositionalCharacterLevelWordTokenizer):
    language_options = ["en", "th"]

    def __init__(self, 
                num_embeddings: int,
                padding_idx: int=0,
                max_positional: int=10,
                language: str="en"):
        assert language in self.language_options

        super().__init__(num_embeddings, padding_idx, max_positional)
        self.word2syllable = Word2Syllable(language)

    def positionize(self, chars: list[str]):
        syllables = self.word2syllable("".join(chars))
        positions = []
        for i, syllable in enumerate(syllables):
            for _ in syllable:
                positions.append(min(i, self.max_positional - 1))
        return chars, positions


class PrecisePositionalCharacterLevelWordTokenizer(PositionalCharacterLevelWordTokenizer):
    def positionize(self, chars: list[str]):
        positions = [min(i, self.max_positional - 1) for i in range(len(chars))]
        return chars, positions


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


class CorpusBasedTokenizer(TextTokenizer):
    def __init__(self,
                 local_dir: str="corpus_based_tokenizer",
                 num_embeddings: int=None, 
                 padding_idx: int=0,
                 input_word: bool=False):

        super().__init__(num_embeddings, padding_idx, input_word)
        self.local_dir = local_dir
        self.token2id = None
        self.load()

    def numerize(self, token: str):
        """ Convert a given token into a number """
        assert self.token2id is not None, "Please fit corpus first"

        number = self.token2id.get(token, self.token2id["<UNK>"])
        return number

    def save(self):
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)

        joblib.dump(self.token2id, self.local_dir + "/token2id.pkl")

    def load(self):
        if os.path.exists(self.local_dir + "/token2id.pkl"):
            self.token2id = joblib.load(self.local_dir + "/token2id.pkl")

    def fit(self, corpus: list[str]):
        tokens_freq = {}
        for string in tqdm(corpus):
            tokens = self.tokenize(string)
            for token in tokens:
                tokens_freq[token] = tokens_freq.get(token, 0) + 1

        tokens_freq = sorted(tokens_freq.items(), key=lambda x: x[1], reverse=True)
        if self.num_embeddings is not None:
            tokens_freq = tokens_freq[:self.num_embeddings - len(self.special_tokens)]
        else:
            self.num_embeddings = len(tokens_freq) + len(self.special_tokens)

        tokens = self.special_tokens + [token for token, _ in tokens_freq]

        self.token2id = {token: i for i, token in enumerate(tokens)}

        self.save()


class WordTokenizer(CorpusBasedTokenizer):
    def __init__(self,
                 local_dir: str="word_tokenizer",
                 num_embeddings: int=None, 
                 padding_idx: int=0,
                 input_word: bool=False):

        super().__init__(local_dir, num_embeddings, padding_idx, input_word)

    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        return word_tokenize(string) if not self.input_word else [string]
