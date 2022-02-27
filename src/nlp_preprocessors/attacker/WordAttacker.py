import os
import gzip
import shutil
import numpy as np
import urllib.request
import nlpaug.augmenter.word as naw

from abc import abstractmethod
from progressist import ProgressBar


BASE_DIR = os.path.dirname(__file__)


def download_embedding():
    download_url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    download_dir = BASE_DIR + "/GoogleNews-vectors-negative300.bin"

    # Download
    bar = ProgressBar(template="|{animation}| {done:B}/{total:B}")
    _ = urllib.request.urlretrieve(download_url, download_dir + ".gz", reporthook=bar.on_urlretrieve)
     # Unzip file
    with gzip.open(download_dir + ".gz", "rb") as f_in:
        with open(download_dir, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    # Remove zip file
    os.remove(download_dir + ".gz")


class BaseWordAttacker:
    @abstractmethod
    def get_pert_sentence(self, tokens, word_id):
        pass

    @staticmethod
    def sentence_validation(tokens):
        return len(tokens) > 3

    @staticmethod
    def get_word_id(tokens):
        valid_indices = np.arange(len(tokens))
        word_id = np.random.choice(valid_indices, size=1)[0]
        return word_id

    def __call__(self, sentence=None, tokens=None, word_id=None):
        # Get tokens
        if tokens is None:
            tokens = sentence.split(" ")
        # Validation
        if not self.sentence_validation(tokens):
            return sentence
        # Get word_id
        if word_id is None:
            word_id = self.get_word_id(tokens)
        # Get perturbated sentence
        if word_id is None:
            return sentence
        else:
            pert_sentence = self.get_pert_sentence(tokens, word_id)
            return pert_sentence


class DropWordAttacker(BaseWordAttacker):
    def get_pert_sentence(self, tokens, word_id):
        del tokens[word_id]
        return " ".join(tokens)


class SwapWordAttacker(BaseWordAttacker):
    @staticmethod
    def get_word_id(tokens):
        valid_indices = list(range(len(tokens)))
        valid_indices.remove(len(valid_indices) - 1)
        word_id = np.random.choice(valid_indices, size=1)[0]
        return word_id

    def get_pert_sentence(self, tokens, word_id):
        token1, token2 = tokens[word_id], tokens[word_id + 1]
        tokens[word_id], tokens[word_id + 1] = token2, token1
        return " ".join(tokens)


class InsertWordAttacker:
    def __init__(self, fast_mode=False):
        if fast_mode:
            # Download embedding file if not exists
            if not os.path.exists(BASE_DIR + "/GoogleNews-vectors-negative300.bin"):
                download_embedding()
            self.aug = naw.WordEmbsAug(model_type="word2vec", model_path=BASE_DIR + "/GoogleNews-vectors-negative300.bin", action="insert")
        else:
            self.aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="insert")

    def __call__(self, sentence):
        return self.aug.augment(sentence)


class SubstituteWordAttacker:
    def __init__(self, fast_mode=False):
        if fast_mode:
            self.aug = naw.SynonymAug(aug_src="wordnet")
        else:
            self.aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="substitute")

    def __call__(self, sentence):
        return self.aug.augment(sentence)


class RandomWordAttacker:
    def __init__(self, insert_p=0.25, drop_p=0.25, swap_p=0.25, substitute_p=0.25):
        self.insert_p = insert_p
        self.drop_p = drop_p
        self.swap_p = swap_p
        self.substitute_p = substitute_p
        self.insert_attacker = InsertWordAttacker()
        self.drop_attacker = DropWordAttacker()
        self.swap_attacker = SwapWordAttacker()
        self.substitute_attacker = SubstituteWordAttacker()

    def __call__(self, sentence):
        # Get attacking methods
        attack_method = np.random.choice(["insert", "drop", "swap", "substitute"], size=1, replace=True, p=[self.insert_p, self.drop_p, self.swap_p, self.substitute_p])[0]
        # Attack
        if attack_method == "insert":
            pert_sentence = self.insert_attacker(sentence)
        elif attack_method == "drop":
            pert_sentence = self.drop_attacker(sentence)
        elif attack_method == "swap":
            pert_sentence = self.swap_attacker(sentence)
        elif attack_method == "substitute":
            pert_sentence = self.substitute_attacker(sentence)
        return pert_sentence