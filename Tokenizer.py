import hashlib

from typing import Union
from abc import abstractmethod
from nltk import word_tokenize


class BaseTokenizer:
    special_tokens = ["<PAD>", "<CLS>", "<SEP>", "<MASK>", "<UNK>"]

    def __init__(self, 
                num_embeddings: int, 
                padding_idx: int=0):

        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx

    def __call__(self, sentences: Union[list[str], str]):
        if isinstance(sentences, list):
            return [[self.numerize(token) for token in self.tokenize(sentence)] for sentence in sentences]
        else:
            return [self.numerize(token) for token in self.tokenize(sentences)]

    @abstractmethod
    def tokenize(self, sentence: str):
        """ Convert a given sentence into a sequence of tokens """
        pass

    @abstractmethod
    def numerize(self, string: str):
        """ Convert a given string into a number """
        pass


class LearningBasedTokenizer(BaseTokenizer):
    @abstractmethod
    def tokenize(self, sentence: str):
        """ Convert a given sentence into a sequence of tokens """
        pass

    @abstractmethod
    def numerize(self, string: str):
        """ Convert a given string into a number """
        pass


class HashingBasedTokenizer(BaseTokenizer):
    def numerize(self, string: str):
        """ Convert a given string into a number """
        hash_number = int(hashlib.sha3_224(bytes(string, "utf8")).hexdigest(), 16) % self.num_embeddings
        hash_number = max(hash_number, self.padding_idx + len(self.special_tokens))
        return hash_number

    def collision_test(self, strings: list):
        """ Return True when no collision in the given strings, and False otherwise """
        cache = {}
        for string in strings:
            hash_number = self.numerize(string)
            if hash_number not in cache:
                cache[hash_number] = string
            else:
                return False
        return True

    @abstractmethod
    def tokenize(self, sentence: str):
        """ Convert a given sentence into a sequence of tokens """
        pass


class CharacterHashWordTokenizer(HashingBasedTokenizer):
    def tokenize(self, sentence: str):
        """ Convert a given sentence into a sequence of tokens """
        words = word_tokenize(sentence)
        tokens = [list(word) for word in words]
        return tokens

    def collision_test(self, strings: list):
        """ Return True when no collision in the given strings, and False otherwise """
        cache = {}
        for string in strings:
            hash_numbers = self.numerize(string)
            hash_numbers = tuple(sorted(hash_numbers))
            if hash_numbers not in cache:
                cache[hash_numbers] = string
            else:
                return False
        return True

    def numerize(self, chars: list):
        """ Convert a given list of strings into a list of numbers """
        sub = super()
        return [sub.numerize(char) for char in chars]


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