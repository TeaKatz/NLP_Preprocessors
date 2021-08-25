from nltk import word_tokenize

from .BaseTokenizer import CorpusBasedTokenizer


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