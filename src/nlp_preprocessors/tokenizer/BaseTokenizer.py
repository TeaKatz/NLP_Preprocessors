import os
import joblib
import hashlib

from abc import abstractmethod
from .utilities import LocalitySensitiveHashing


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
        for token in corpus:
            tokens_freq[token] = tokens_freq.get(token, 0) + 1

        tokens_freq = sorted(tokens_freq.items(), key=lambda x: x[1], reverse=True)
        if self.num_embeddings is not None:
            tokens_freq = tokens_freq[:self.num_embeddings - len(self.special_tokens)]
        else:
            self.num_embeddings = len(tokens_freq) + len(self.special_tokens)

        tokens = self.special_tokens + [token for token, _ in tokens_freq]

        self.token2id = {token: i for i, token in enumerate(tokens)}

        self.save()