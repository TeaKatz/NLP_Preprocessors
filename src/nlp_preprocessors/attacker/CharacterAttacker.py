import math
import numpy as np


class EngCharacterAttacker:
    eng_characters = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", 
                      "a", "s", "d", "f", "g", "h", "j", "k", "l", "z",
                      "x", "c", "v", "b", "n", "m"]

    def __init__(self, words_num=0.1, insert_p=0.25, drop_p=0.25, swap_p=0.25, substitute_p=0.25):
        """
        words_num: Number (integer) or percentage (float) of words to attack in the given sentence
        insert_p: Probability (float) of attacking by inserting
        drop_p: Probability (float) of attacking by dropping
        swap_p: Probability (float) of attacking by swapping
        substitute_p: Probability (float) of attacking by substitution
        """
        self.words_num = words_num
        self.insert_p = insert_p
        self.drop_p = drop_p
        self.swap_p = swap_p
        self.substitute_p = substitute_p

    def insert(self, word):
        word = list(word)

        insert_id = np.random.choice(len(word), size=1)[0]
        insert_char = np.random.choice(self.eng_characters, size=1)[0]

        word.insert(insert_id, insert_char)
        word = "".join(word)

        return word

    def drop(self, word):
        word = list(word)

        drop_id = np.random.choice(len(word), size=1)[0]

        del word[drop_id]
        word = "".join(word)
        
        return word

    def swap(self, word):
        word = list(word)

        swap_id = np.random.choice(len(word) - 1, size=1)[0]

        char1, char2 = word[swap_id], word[swap_id + 1]
        word[swap_id], word[swap_id + 1] = char2, char1
        word = "".join(word)
        
        return word

    def substitute(self, word):
        word = list(word)

        substitute_id = np.random.choice(len(word) - 1, size=1)[0]
        substitute_char = np.random.choice(self.eng_characters, size=1)[0]

        word[substitute_id] = substitute_char
        word = "".join(word)
        
        return word

    def __call__(self, sentence):
        words = sentence.split(" ")

        valid_indices = [id for id in np.arange(len(words)) if len(words[id]) >= 3]
        if isinstance(self.words_num, int):
            attack_num = min(self.words_num, len(valid_indices))
        else:
            attack_num = math.ceil(self.words_num * len(valid_indices))
        attack_indices = np.random.choice(valid_indices, size=attack_num, replace=False)
        attack_methods = np.random.choice(["insert", "drop", "swap", "substitute"], size=attack_num, replace=True, p=[self.insert_p, self.drop_p, self.swap_p, self.substitute_p])

        for attack_id, attack_method in zip(attack_indices, attack_methods):
            if attack_method == "insert":
                words[attack_id] = self.insert(words[attack_id])
            elif attack_method == "drop":
                words[attack_id] = self.drop(words[attack_id])
            elif attack_method == "swap":
                words[attack_id] = self.swap(words[attack_id])
            else:
                words[attack_id] = self.substitute(words[attack_id])
        sentence = " ".join(words)

        return sentence