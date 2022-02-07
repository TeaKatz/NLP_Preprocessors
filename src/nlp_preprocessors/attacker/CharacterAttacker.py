import math
import torch
import numpy as np
from abc import abstractmethod
from torch.nn import CosineSimilarity


eng_characters = [
    "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", 
    "a", "s", "d", "f", "g", "h", "j", "k", "l", "z",
    "x", "c", "v", "b", "n", "m"
]
nearby_characters = {
    "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk", "i": "uojkl", "o": "ipkl", "p": "ol",
    "a": "qwszx", "s": "qweadzxc", "d": "wersfxcv", "f": "ertdgcvb", "g": "rtyfhvbn", "h": "tyugjbnm", "j": "yuihknm", "k": "uiojlm", "l": "iopk", 
    "z": "asx", "x": "assdzc", "c": "sdfxv", "v": "dfgcb", "b": "fghvn", "n": "ghjbm", "m": "hjkn"
}


def word_validation(word):
    return len(word) >= 3 and len([char for char in word if char in eng_characters]) > 0


class BaseCharacterAttacker:
    def __init__(self, keyboard_constrain=False):
        self.keyboard_constrain = keyboard_constrain

    @staticmethod
    def get_word_id(tokens):
        valid_indices = [id for id in np.arange(len(tokens)) if word_validation(tokens[id])]
        if len(valid_indices) == 0:
            return None
        word_id = np.random.choice(valid_indices, size=1, replace=False)[0]
        return word_id

    @staticmethod
    def get_char_id(word):
        valid_indices = [id for id, char in enumerate(word) if char in eng_characters]
        if len(valid_indices) == 0:
            return None
        char_id = np.random.choice(valid_indices, size=1)[0]
        return char_id

    @abstractmethod
    def get_pert_words(self, word, char_id, return_all_possible=False):
        pass

    def __call__(self, sentence=None, tokens=None, word_id=None, char_id=None, return_all_possible=False, return_tokens=False):
        # Get tokens
        if tokens is None:
            tokens = sentence.split(" ")
        # Get word_id
        if word_id is None:
            word_id = self.get_word_id(tokens)
            if word_id is None:
                if return_tokens:
                    return tokens
                else:
                    return " ".join(tokens)
        # Get target perturbation word
        target_word = list(tokens[word_id])
        # Get char_id
        if char_id is None:
            char_id = self.get_char_id(target_word)
            if char_id is None:
                if return_tokens:
                    return tokens
                else:
                    return " ".join(tokens)
        # Get perturbated words
        pert_words = self.get_pert_words(target_word, char_id, return_all_possible)
        # Get perturbated outputs
        pert_outputs = []
        for pert_word in pert_words:
            pert_tokens = tokens.copy()
            pert_tokens[word_id] = pert_word
            if return_tokens:
                pert_outputs.append(pert_tokens)
            else:
                pert_outputs.append(" ".join(pert_tokens))
        return pert_outputs if return_all_possible else pert_outputs[0]


class InsertCharacterAttacker(BaseCharacterAttacker):
    def get_pert_words(self, word, char_id, return_all_possible=False):
        if self.keyboard_constrain:
            target_char = word[char_id]
            if target_char in nearby_characters:
                if return_all_possible:
                    insert_chars = list(nearby_characters[target_char])
                else:
                    insert_chars = np.random.choice(list(nearby_characters[target_char]), size=1)
            else:
                insert_chars = [target_char]
        else:
            if return_all_possible:
                insert_chars = eng_characters
            else:
                insert_chars = np.random.choice(eng_characters, size=1)
        pert_words = []
        for insert_char in insert_chars:
            t_word = word.copy()
            t_word.insert(char_id, insert_char)
            pert_word = "".join(t_word)
            pert_words.append(pert_word)
            if not return_all_possible:
                break
        return pert_words


class DropCharacterAttacker(BaseCharacterAttacker):
    def get_pert_words(self, word, char_id, return_all_possible=False):
        del word[char_id]
        pert_words = ["".join(word)]
        return pert_words


class SwapCharacterAttacker(BaseCharacterAttacker):
    @staticmethod
    def get_char_id(word):
        valid_indices = [id for id, char in enumerate(word) if char in eng_characters]
        if len(word) - 1 in valid_indices:
            valid_indices.remove(len(word) - 1)
        if len(valid_indices) == 0:
            return None
        char_id = np.random.choice(valid_indices, size=1)[0]
        return char_id

    def get_pert_words(self, word, char_id, return_all_possible=False):
        char1, char2 = word[char_id], word[char_id + 1]
        word[char_id], word[char_id + 1] = char2, char1
        pert_words = ["".join(word)]
        return pert_words


class SubstituteCharacterAttacker(BaseCharacterAttacker):
    def get_pert_words(self, word, char_id, return_all_possible=False):
        if self.keyboard_constrain:
            target_char = word[char_id]
            if target_char in nearby_characters:
                if return_all_possible:
                    substitute_chars = list(nearby_characters[target_char])
                else:
                    substitute_chars = np.random.choice(list(nearby_characters[target_char]), size=1)
            else:
                substitute_chars = [target_char]
        else:
            if return_all_possible:
                substitute_chars = eng_characters
            else:
                substitute_chars = np.random.choice(eng_characters, size=1)
        pert_words = []
        for substitute_char in substitute_chars:
            t_word = word.copy()
            t_word[char_id] = substitute_char
            pert_word = "".join(t_word)
            pert_words.append(pert_word)
            if not return_all_possible:
                break
        return pert_words


class RandomCharacterAttacker:
    def __init__(self, insert_p=0.25, drop_p=0.25, swap_p=0.25, substitute_p=0.25, keyboard_constrain=False):
        self.insert_p = insert_p
        self.drop_p = drop_p
        self.swap_p = swap_p
        self.substitute_p = substitute_p
        self.insert_attacker = InsertCharacterAttacker(keyboard_constrain)
        self.drop_attacker = DropCharacterAttacker(keyboard_constrain)
        self.swap_attacker = SwapCharacterAttacker(keyboard_constrain)
        self.substitute_attacker = SubstituteCharacterAttacker(keyboard_constrain)

    def __call__(self, sentence=None, tokens=None, perturb_num=1, word_indices=None, attack_methods=None, char_indices=None, return_tokens=False):
        if sentence is not None:
            assert isinstance(sentence, str)
        if tokens is not None:
            assert isinstance(tokens, list)
        
        if perturb_num is None:
            if sentence is not None:
                return sentence
            elif tokens is not None:
                return tokens

        assert isinstance(perturb_num, int) or isinstance(perturb_num, float), "perturb_num must be int or float."
        if word_indices is not None:
            assert isinstance(word_indices, list), "word_indices must be a list."
            if attack_methods is not None:
                assert isinstance(attack_methods, list), "attack_methods must be a list."
                assert len(attack_methods) == len(word_indices), "length of attack_methods and word_indices must equal."
            if char_indices is not None:
                assert isinstance(char_indices, list), "char_indices must be a list."
                assert len(char_indices) == len(word_indices), "length of char_indices and word_indices must equal."

        # Get tokens
        if tokens is None:
            tokens = sentence.split(" ")
        # Get word indices
        if word_indices is None:
            valid_indices = [id for id in np.arange(len(tokens)) if word_validation(tokens[id])]
            if len(valid_indices) == 0:
                if return_tokens:
                    return tokens
                else:
                    return " ".join(tokens)
            if isinstance(perturb_num, int):
                perturb_num = min(perturb_num, len(valid_indices))
            else:
                perturb_num = math.ceil(perturb_num * len(valid_indices))
            word_indices = np.random.choice(valid_indices, size=perturb_num, replace=False)
        # Get attacking methods
        if attack_methods is None:
            attack_methods = np.random.choice(["insert", "drop", "swap", "substitute"], size=len(word_indices), replace=True, p=[self.insert_p, self.drop_p, self.swap_p, self.substitute_p])
        # Get character indices
        if char_indices is None:
            char_indices = [None] * len(word_indices)
        # Attack
        pert_tokens = tokens
        for word_id, attack_method, char_id in zip(word_indices, attack_methods, char_indices):
            if attack_method == "insert":
                pert_tokens = self.insert_attacker(tokens=pert_tokens, word_id=word_id, char_id=char_id, return_tokens=True)
            elif attack_method == "drop":
                pert_tokens = self.drop_attacker(tokens=pert_tokens, word_id=word_id, char_id=char_id, return_tokens=True)
            elif attack_method == "swap":
                pert_tokens = self.swap_attacker(tokens=pert_tokens, word_id=word_id, char_id=char_id, return_tokens=True)
            elif attack_method == "substitute":
                pert_tokens = self.substitute_attacker(tokens=pert_tokens, word_id=word_id, char_id=char_id, return_tokens=True)
        if return_tokens:
            return pert_tokens
        else:
            return " ".join(pert_tokens)


class UnsupervisedAdversarialCharacterAttacker:
    def __init__(self, sentences_embedding, keyboard_constrain=False):
        self.sentences_embedding = sentences_embedding
        self.insert_attacker = InsertCharacterAttacker(keyboard_constrain)
        self.drop_attacker = DropCharacterAttacker(keyboard_constrain)
        self.swap_attacker = SwapCharacterAttacker(keyboard_constrain)
        self.substitute_attacker = SubstituteCharacterAttacker(keyboard_constrain)

    def __call__(self, sentence1, sentence2=None, perturb_num=1):
        tokens1 = sentence1.split(" ")
        if sentence2 is not None:
            tokens2 = sentence2.split(" ")

        # Initial perturbated sentences
        pert_sentence1 = sentence1
        if sentence2 is not None:
            pert_sentence2 = sentence2

        # Get original sentence embeddings
        with torch.no_grad():
            sent_embedding1 = self.sentences_embedding([sentence1])         # (1, hidden_size)
            if sentence2 is not None:
                sent_embedding2 = self.sentences_embedding([sentence2])     # (1, hidden_size)
                # Calculate sentences similarity
                ori_sents_sim = CosineSimilarity(dim=-1)(sent_embedding1, sent_embedding2)  # (1, )

            # Search the most important word (token)
            t_pert_sentences1 = [" ".join(tokens1[:i] + tokens1[i + 1:]) if word_validation(word) else sentence1 for i, word in enumerate(tokens1)]
            if sentence2 is not None:
                t_pert_sentences2 = [" ".join(tokens2[:i] + tokens2[i + 1:]) if word_validation(word) else sentence2 for i, word in enumerate(tokens2)]
                t_pert_embedding1 = self.sentences_embedding(t_pert_sentences1)     # (tokens1_size, hidden_size)
                t_pert_embedding2 = self.sentences_embedding(t_pert_sentences2)     # (tokens2_size, hidden_size)
                # Calculate sentences similarity
                pert_sents_sim = CosineSimilarity(dim=-1)(t_pert_embedding1.unsqueeze(1), t_pert_embedding2.unsqueeze(0))   # (tokens1_size, tokens2_size)
                # Get the most importance token
                diff_sents_sim = torch.abs(pert_sents_sim - ori_sents_sim)
                flatten_ids = torch.argsort(diff_sents_sim.flatten(), descending=True)
                word1_ids = (flatten_ids / diff_sents_sim.size(1)).long()
                word2_ids = (flatten_ids % diff_sents_sim.size(1)).long()
            else:
                t_pert_embedding1 = self.sentences_embedding(t_pert_sentences1)
                # Get the most importance token
                diff_sents_embedding = torch.mean(torch.abs(t_pert_embedding1 - sent_embedding1), dim=-1)     # (tokens1_size, )
                word1_ids = torch.argsort(diff_sents_embedding, descending=True)

            for i in range(perturb_num):
                word1_id = word1_ids[i]
                word2_id = word2_ids[i]
                # Search the most important character
                t_pert_sentences1 = [self.drop_attacker(pert_sentence1, word_id=word1_id, char_id=char_id) for char_id in range(len(pert_sentence1.split(" ")[word1_id]))]
                if sentence2 is not None:
                    t_pert_sentences2 = [self.drop_attacker(pert_sentence2, word_id=word2_id, char_id=char_id) for char_id in range(len(pert_sentence2.split(" ")[word2_id]))]
                    t_pert_embedding1 = self.sentences_embedding(t_pert_sentences1)     # (char1_size, hidden_size)
                    t_pert_embedding2 = self.sentences_embedding(t_pert_sentences2)     # (char2_size, hidden_size)
                    # Calculate sentences similarity
                    pert_sents_sim = CosineSimilarity(dim=-1)(t_pert_embedding1.unsqueeze(1), t_pert_embedding2.unsqueeze(0))   # (char1_size, char2_size)
                    # Get the most importance token
                    diff_sents_sim = torch.abs(pert_sents_sim - ori_sents_sim)
                    flatten_id = torch.argmax(diff_sents_sim)
                    char1_id = int(flatten_id / diff_sents_sim.size(1))
                    char2_id = flatten_id % diff_sents_sim.size(1)
                else:
                    t_pert_embedding1 = self.sentences_embedding(t_pert_sentences1)
                    # Get the most importance token
                    diff_sents_embedding = torch.mean(torch.abs(t_pert_embedding1 - sent_embedding1), dim=-1)     # (char1_size, )
                    char1_id = torch.argmax(diff_sents_embedding)

                # Get perturbated sentences
                t_pert_sentences1 = self.insert_attacker(pert_sentence1, word_id=word1_id, char_id=char1_id, return_all_possible=True)
                t_pert_sentences1 += self.drop_attacker(pert_sentence1, word_id=word1_id, char_id=char1_id, return_all_possible=True)
                t_pert_sentences1 += self.swap_attacker(pert_sentence1, word_id=word1_id, char_id=char1_id, return_all_possible=True)
                t_pert_sentences1 += self.substitute_attacker(pert_sentence1, word_id=word1_id, char_id=char1_id, return_all_possible=True)
                if sentence2 is not None:
                    t_pert_sentences2 = self.insert_attacker(pert_sentence2, word_id=word2_id, char_id=char2_id, return_all_possible=True)
                    t_pert_sentences2 += self.drop_attacker(pert_sentence2, word_id=word2_id, char_id=char2_id, return_all_possible=True)
                    t_pert_sentences2 += self.swap_attacker(pert_sentence2, word_id=word2_id, char_id=char2_id, return_all_possible=True)
                    t_pert_sentences2 += self.substitute_attacker(pert_sentence2, word_id=word2_id, char_id=char2_id, return_all_possible=True)
                    t_pert_embedding1 = self.sentences_embedding(t_pert_sentences1)     # (sent1_size, hidden_size)
                    t_pert_embedding2 = self.sentences_embedding(t_pert_sentences2)     # (sent2_size, hidden_size)
                    # Calculate sentences similarity
                    pert_sents_sim = CosineSimilarity(dim=-1)(t_pert_embedding1.unsqueeze(1), t_pert_embedding2.unsqueeze(0))   # (sent1_size, sent2_size)
                    # Get the most importance token
                    diff_sents_sim = torch.abs(pert_sents_sim - ori_sents_sim)
                    flatten_id = torch.argmax(diff_sents_sim)
                    sent1_id = int(flatten_id / diff_sents_sim.size(1))
                    sent2_id = flatten_id % diff_sents_sim.size(1)
                else:
                    t_pert_embedding1 = self.sentences_embedding(t_pert_sentences1)
                    # Get the most importance token
                    diff_sents_embedding = torch.mean(torch.abs(t_pert_embedding1 - sent_embedding1), dim=-1)     # (sent1_size, )
                    sent1_id = torch.argmax(diff_sents_embedding)

                # Update perturbated sentence output
                pert_sentence1 = t_pert_sentences1[sent1_id]
                if sentence2 is not None:
                    pert_sentence2 = t_pert_sentences2[sent2_id]

        if sentence2 is not None:
            return pert_sentence1, pert_sentence2
        else:
            return pert_sentence1


class SupervisedAdversarialCharacterAttacker:
    def __init__(self, model_prediction, keyboard_constrain=False):
        self.model_prediction = model_prediction
        self.insert_attacker = InsertCharacterAttacker(keyboard_constrain)
        self.drop_attacker = DropCharacterAttacker(keyboard_constrain)
        self.swap_attacker = SwapCharacterAttacker(keyboard_constrain)
        self.substitute_attacker = SubstituteCharacterAttacker(keyboard_constrain)

    def __call__(self, sentence1, sentence2=None, target=None, objective=None, perturb_num=1):
        tokens1 = sentence1.split(" ")
        if sentence2 is not None:
            tokens2 = sentence2.split(" ")

        # Initial perturbated sentences
        pert_sentence1 = sentence1
        if sentence2 is not None:
            pert_sentence2 = sentence2

        # Get original sentence embeddings
        with torch.no_grad():
            # Search the most important word (token)
            t_pert_sentences1 = [" ".join(tokens1[:i] + tokens1[i + 1:]) if word_validation(word) else sentence1 for i, word in enumerate(tokens1)]
            if sentence2 is not None:
                t_pert_sentences2 = [" ".join(tokens2[:i] + tokens2[i + 1:]) if word_validation(word) else sentence2 for i, word in enumerate(tokens2)]
                t_pert_sentences_pair = [[sentence1, sentence2] for sentence1 in t_pert_sentences1 for sentence2 in t_pert_sentences2]
                t_pert_pred = self.model_prediction(*t_pert_sentences_pair)     # (tokens1_size * tokens2_size, output_size)
                # Get the most importance token
                loss = objective(t_pert_pred, target)
                flatten_ids = torch.argsort(loss, descending=True)
                word1_ids = (flatten_ids / len(t_pert_sentences2)).long()
                word2_ids = (flatten_ids % len(t_pert_sentences2)).long()
            else:
                t_pert_pred = self.model_prediction(t_pert_sentences1)
                # Get the most importance token
                loss = objective(t_pert_pred, target)
                word1_ids = torch.argsort(loss, descending=True)

            for i in range(perturb_num):
                word1_id = word1_ids[i]
                word2_id = word2_ids[i]
                # Search the most important character
                t_pert_sentences1 = [self.drop_attacker(pert_sentence1, word_id=word1_id, char_id=char_id) for char_id in range(len(pert_sentence1.split(" ")[word1_id]))]
                if sentence2 is not None:
                    t_pert_sentences2 = [self.drop_attacker(pert_sentence2, word_id=word2_id, char_id=char_id) for char_id in range(len(pert_sentence2.split(" ")[word2_id]))]
                    t_pert_pred = self.model_prediction(*t_pert_sentences_pair)     # (char1_size * char2_size, output_size)
                    # Get the most importance token
                    loss = objective(t_pert_pred, target)
                    flatten_id = torch.argmax(loss)
                    char1_id = int(flatten_id / len(t_pert_sentences2))
                    char2_id = flatten_id % len(t_pert_sentences2)
                else:
                    t_pert_pred = self.model_prediction(t_pert_sentences1)
                    # Get the most importance token
                    loss = objective(t_pert_pred, target)
                    char1_id = torch.argmax(loss)

                # Get perturbated sentences
                t_pert_sentences1 = self.insert_attacker(pert_sentence1, word_id=word1_id, char_id=char1_id, return_all_possible=True)
                t_pert_sentences1 += self.drop_attacker(pert_sentence1, word_id=word1_id, char_id=char1_id, return_all_possible=True)
                t_pert_sentences1 += self.swap_attacker(pert_sentence1, word_id=word1_id, char_id=char1_id, return_all_possible=True)
                t_pert_sentences1 += self.substitute_attacker(pert_sentence1, word_id=word1_id, char_id=char1_id, return_all_possible=True)
                if sentence2 is not None:
                    t_pert_sentences2 = self.insert_attacker(pert_sentence2, word_id=word2_id, char_id=char2_id, return_all_possible=True)
                    t_pert_sentences2 += self.drop_attacker(pert_sentence2, word_id=word2_id, char_id=char2_id, return_all_possible=True)
                    t_pert_sentences2 += self.swap_attacker(pert_sentence2, word_id=word2_id, char_id=char2_id, return_all_possible=True)
                    t_pert_sentences2 += self.substitute_attacker(pert_sentence2, word_id=word2_id, char_id=char2_id, return_all_possible=True)
                    t_pert_pred = self.model_prediction(*t_pert_sentences_pair)     # (sent1_size * sent2_size, output_size)
                    # Get the most importance token
                    loss = objective(t_pert_pred, target)
                    flatten_id = torch.argmax(loss)
                    sent1_id = int(flatten_id / len(t_pert_sentences2))
                    sent2_id = flatten_id % len(t_pert_sentences2)
                else:
                    t_pert_pred = self.model_prediction(t_pert_sentences1)
                    # Get the most importance token
                    loss = objective(t_pert_pred, target)
                    sent1_id = torch.argmax(loss)

                # Update perturbated sentence output
                pert_sentence1 = t_pert_sentences1[sent1_id]
                if sentence2 is not None:
                    pert_sentence2 = t_pert_sentences2[sent2_id]

        if sentence2 is not None:
            return pert_sentence1, pert_sentence2
        else:
            return pert_sentence1