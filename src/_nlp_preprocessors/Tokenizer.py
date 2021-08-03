import hashlib

from typing import Union
from abc import abstractmethod
from nltk import word_tokenize

from .Utilities import Word2Syllable, word2ngram, word2skipngram, LocalitySensitiveHashing


class BaseTokenizer:
    special_tokens = ["<PAD>", "<CLS>", "<SEP>", "<MASK>", "<UNK>"]

    def __init__(self, 
                num_embeddings: int, 
                padding_idx: int=0,
                input_word: bool=False):

        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.input_word = input_word

    def __call__(self, strings: Union[list[str], str]):
        if isinstance(strings, list):
            return [[self.numerize(token) for token in self.tokenize(string)] for string in strings]
        else:
            return [self.numerize(token) for token in self.tokenize(strings)]

    @abstractmethod
    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        pass

    @abstractmethod
    def numerize(self, token: str):
        """ Convert a given token into a number """
        pass


class HashingBasedTokenizer(BaseTokenizer):
    def numerize(self, token: str):
        """ Convert a given token into a number """
        hash_number = int(hashlib.sha3_224(bytes(token, "utf8")).hexdigest(), 16) % self.num_embeddings
        hash_number = max(hash_number, self.padding_idx + len(self.special_tokens))
        return hash_number

    @abstractmethod
    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        pass


class LocalitySensitiveHashingBasedTokenizer(BaseTokenizer):
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

    @abstractmethod
    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        pass


class WordTokenizer(HashingBasedTokenizer):
    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        return word_tokenize(string) if not self.input_word else [string]


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


class CorpusBasedTokenizer(BaseTokenizer):
    @abstractmethod
    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        pass

    @abstractmethod
    def numerize(self, token: str):
        """ Convert a given token into a number """
        pass


# class _BaseTokenizer:
#     def __init__(self, 
#                 max_vocabs: int=None,
#                 min_freq: int=None,
#                 pad_to_length: Union[None, str, int]=None,
#                 truncate: bool=False,
#                 pad_token_id: int=0,
#                 cls_token_id: int=1,
#                 sep_token_id: int=2,
#                 mask_token_id: int=3,
#                 return_padding_mask: bool=False):

#         self.max_vocabs = max_vocabs
#         self.min_freq = min_freq
#         self.padding = TokenPadding(pad_to_length, truncate, pad_token_id, return_padding_mask)
#         self.pad_token_id = pad_token_id
#         self.cls_token_id = cls_token_id
#         self.sep_token_id = sep_token_id
#         self.mask_token_id = mask_token_id
#         self.vocabs = []
#         self.vocab2id = {}
#         self.id2vocab = {}

#     def __call__(self, texts: List[str], start_token=False, end_token=False) -> Dict:
#         token_ids = np.asarray([self.text2ids(text, start_token=start_token, end_token=end_token) for text in texts])
#         return self.padding(token_ids)

#     @abstractmethod
#     def tokenizer(self, text):
#         pass

#     def text2ids(self, text, start_token=False, end_token=False):
#         tokens = self.text2tokens(text, start_token=start_token, end_token=end_token)

#         token_ids = [self.vocab2id.get(token, self.unk_token_id) for token in tokens]
#         return token_ids

#     def text2tokens(self, text, start_token=False, end_token=False):
#         tokens = self.tokenizer(text)
#         if start_token:
#             tokens = [self.cls_token] + tokens
#         if end_token:
#             tokens = tokens + [self.sep_token]
#         return tokens

#     def ids2text(self, ids, remove_special_token=True):
#         tokens = self.ids2tokens(ids, remove_special_token=remove_special_token)
#         text = "".join(tokens)
#         return text

#     def ids2tokens(self, ids, remove_special_token=True):
#         tokens = []
#         for i in ids:
#             if remove_special_token and i in [self.pad_token_id, self.cls_token_id, self.sep_token_id, self.mask_token_id, self.unk_token_id]:
#                 continue
#             tokens.append(self.id2vocab[i])
#         return tokens

#     def save(self, save_dir):
#         with open(save_dir, "w") as f:
#             [f.write(vocab + "\n") for vocab in self.vocabs]

#     def load(self, load_dir):
#         with open(load_dir, "r") as f:
#             self.vocabs = f.read()[:-1].split("\n")
#             # Get vocab2id and id2vocab
#             self.vocab2id[self.pad_token] = self.pad_token_id
#             self.vocab2id[self.cls_token] = self.cls_token_id
#             self.vocab2id[self.sep_token] = self.sep_token_id
#             self.vocab2id[self.mask_token] = self.mask_token_id

#             self.vocab2id.update({token: i + len(self.vocab2id) for i, token in enumerate(self.vocabs)})

#             self.vocab2id[self.unk_token] = self.unk_token_id
#             self.id2vocab = {i: token for token, i in self.vocab2id.items()}

#     def fit(self, corpus: List, initial_vocabs: List=None):
#         # Get tokens frequency
#         token_freq = {}
#         for text in tqdm(corpus):
#             tokens = self.tokenizer(text)
#             for token in tokens:
#                 token_freq[token] = token_freq.get(token, 0) + 1

#         # Get vocabs
#         if self.max_vocabs is not None and self.min_freq is not None:
#             sorted_token_freq = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
#             self.vocabs = [token for token, freq in sorted_token_freq[:self.max_vocabs] if freq >= self.min_freq]
#         elif self.max_vocabs is not None:
#             sorted_token_freq = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
#             self.vocabs = [token for token, _ in sorted_token_freq[:self.max_vocabs]]
#         elif self.min_freq is not None:
#             self.vocabs = [token for token, freq in token_freq.items() if freq >= self.min_freq]
#         else:
#             self.vocabs = list(token_freq.keys())

#         # Get vocab2id and id2vocab
#         self.vocab2id[self.pad_token] = self.pad_token_id
#         self.vocab2id[self.cls_token] = self.cls_token_id
#         self.vocab2id[self.sep_token] = self.sep_token_id
#         self.vocab2id[self.mask_token] = self.mask_token_id

#         if initial_vocabs is not None:
#             self.vocabs = initial_vocabs + self.vocabs
#         self.vocab2id.update({token: i + len(self.vocab2id) for i, token in enumerate(self.vocabs)})

#         self.vocab2id[self.unk_token] = self.unk_token_id
#         self.id2vocab = {i: token for token, i in self.vocab2id.items()}

#     @property
#     def vocab_size(self):
#         return len(self.vocab2id)

#     @property
#     def pad_token(self):
#         return "<PAD>"

#     @property
#     def cls_token(self):
#         return "<CLS>"

#     @property
#     def sep_token(self):
#         return "<SEP>"

#     @property
#     def mask_token(self):
#         return "<MASK>"

#     @property
#     def unk_token(self):
#         return "<UNK>"

#     @property
#     def unk_token_id(self):
#         return len(self.vocabs) + 4