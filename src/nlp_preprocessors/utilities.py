import math
import hashlib
import numpy as np

from nltk.corpus import words
from pythainlp.tokenize import syllable_tokenize
from nltk.tokenize import LegalitySyllableTokenizer


def word2ngram(word, n=3):
    grams = [word[i:i+n] for i in range(len(word) - n + 1)]
    return grams


def word2skipngram(word, n=2):
    skipgrams = ["".join([word[i+(j*2)] for j in range(n)]) for i in range(len(word) - 2 * n + 2)]
    return skipgrams


def shorten_signal(signal, threshold=1e-3, offset=100):
    start_id = 0
    for i in np.arange(signal.shape[0]):
        value = signal[i]
        if abs(value) > threshold:
            start_id = i
            break

    end_id = math.inf
    for i in np.arange(signal.shape[0])[::-1]:
        value = signal[i]
        if abs(value) > threshold:
            end_id = i
            break
            
    signal = signal[start_id - offset:end_id + offset]
    return signal


def array2str(array):
    return str(array).replace("[", "").replace("]", "").replace(" ", "")


class Word2Syllable:
    language_options = ["en", "th"]

    def __init__(self, language="en"):
        assert language in self.language_options

        self.language = language
        self.tokenizer = LegalitySyllableTokenizer(words.words()) if language == "en" else syllable_tokenize

    def __call__(self, word):
        if self.language == "en":
            return self.tokenizer.tokenize(word)
        else:
            return self.tokenizer(word)


class LocalitySensitiveHashing:
    def __init__(self, k, random_seed=0):
        self.k = k
        self.random_seed = random_seed
        
    def __call__(self, string):
        ids = [int(hashlib.sha3_224(bytes(char, "utf8")).hexdigest(), 16) % self.k for char in string]
        
        min_id = self.minhashing(ids)
        return min_id
    
    def minhashing(self, ids):
        binary_vector = self.get_binary_vector(ids)
        
        row_indices = np.arange(self.k)
        np.random.seed(self.random_seed)
        np.random.shuffle(row_indices)
        for row in range(self.k):
            idx = np.where(row_indices == row)[0][0]
            val = binary_vector[idx]
            if val == 1:
                min_id = idx
                break
        return min_id
    
    def get_binary_vector(self, ids):
        binary_vector = np.zeros([self.k], dtype=int)
        for idx in ids:
            binary_vector[idx] = 1
        return binary_vector
