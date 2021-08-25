from nltk import word_tokenize

from .utilities import word2ngram
from .utilities import word2skipngram
from .BaseTokenizer import HashingBasedTokenizer
from .BaseTokenizer import LocalitySensitiveHashingBasedTokenizer


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