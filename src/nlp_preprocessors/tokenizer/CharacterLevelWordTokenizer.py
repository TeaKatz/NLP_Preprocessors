from nltk import word_tokenize

from .BaseTokenizer import HashingBasedTokenizer, CorpusBasedTokenizer


class VocabFreeCharacterLevelWordTokenizer(HashingBasedTokenizer):
    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        words = word_tokenize(string) if not self.input_word else [string]
        tokens = [list(word) for word in words]
        return tokens

    def numerize(self, chars: list[str]):
        """ Convert a given list of tokens into a list of numbers """
        sub = super()
        return [sub.numerize(char) for char in chars]


class CharacterLevelWordTokenizer(CorpusBasedTokenizer):
    def __init__(self,
                 local_dir: str="char_tokenizer",
                 num_embeddings: int=None, 
                 padding_idx: int=0,
                 input_word: bool=False):

        super().__init__(local_dir, num_embeddings, padding_idx, input_word)

    def tokenize(self, string: str):
        """ Convert a given string into a sequence of tokens """
        words = word_tokenize(string) if not self.input_word else [string]
        tokens = [list(word) for word in words]
        return tokens

    def numerize(self, chars: list[str]):
        """ Convert a given list of tokens into a list of numbers """
        sub = super()
        return [sub.numerize(char) for char in chars]