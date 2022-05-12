import math
import torch
import numpy as np
from abc import abstractmethod
from sklearn.cluster import KMeans


eng_characters = [
    "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", 
    "a", "s", "d", "f", "g", "h", "j", "k", "l", "z",
    "x", "c", "v", "b", "n", "m"
]
nearby_characters = {
    "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk", "i": "uojkl", "o": "ipkl", "p": "ol",
    "a": "qwszx", "s": "qweadzxc", "d": "wersfzxcv", "f": "ertdgxcvb", "g": "rtyfhcvbn", "h": "tyugjvbnm", "j": "yuihkbnm", "k": "uiojlnm", "l": "iopkm", 
    "z": "asdx", "x": "sdfzc", "c": "dfgxv", "v": "fghcb", "b": "ghjvn", "n": "hjkbm", "m": "jkln"
}
numeric_characters = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]
nearby_numerics = {
    "q": "12", "w": "123", "e": "234", "r": "345", "t": "456", "y": "567", "u": "678", "i": "789", "o": "890", "p": "90"
}


def word_validation(word, min_word_len=3):
    return len(word) >= min_word_len and len([char for char in word if char in eng_characters]) > 0

def sentence2words(sentence):
    return sentence.split(" ")

def words2sentence(words):
    return " ".join(words)

def word2chars(word):
    return list(word)

def chars2word(chars):
    return "".join(chars)


class BaseCharacterAttacker:
    def __init__(self, keyboard_constrain=False, allow_numeric=False, random_seed=None):
        self.keyboard_constrain = keyboard_constrain
        self.allow_numeric = allow_numeric
        self.random_seed = random_seed

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    @staticmethod
    def get_word_id(tokens, min_word_len=3):
        valid_indices = [id for id in np.arange(len(tokens)) if word_validation(tokens[id], min_word_len)]
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

    def __call__(self, sentence=None, tokens=None, word_id=None, char_id=None, min_word_len=3, return_all_possible=False, return_tokens=False):
        # Get tokens
        if tokens is None:
            tokens = sentence.split(" ")
        # Get word_id
        if word_id is None:
            word_id = self.get_word_id(tokens, min_word_len)
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
        if isinstance(word, str):
            word = list(word)
        word = word.copy()

        if self.keyboard_constrain:
            target_char = word[char_id]
            if target_char in nearby_characters:
                candidate_chars = list(nearby_characters[target_char])
                if self.allow_numeric and target_char in nearby_numerics:
                    candidate_chars += list(nearby_numerics[target_char])

                if return_all_possible:
                    insert_chars = candidate_chars
                else:
                    insert_chars = np.random.choice(candidate_chars, size=1)
            else:
                insert_chars = [target_char]
        else:
            candidate_chars = eng_characters
            if self.allow_numeric:
                candidate_chars += numeric_characters

            if return_all_possible:
                insert_chars = candidate_chars
            else:
                insert_chars = np.random.choice(candidate_chars, size=1)
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
    @staticmethod
    def get_char_id(word):
        if len(word) == 0:
            return None

    def get_pert_words(self, word, char_id, return_all_possible=False):
        if isinstance(word, str):
            word = list(word)
        word = word.copy()

        if len(word) == 0:
            return ["".join(word)]

        del word[char_id]
        pert_words = ["".join(word)]
        return pert_words


class SwapCharacterAttacker(BaseCharacterAttacker):
    @staticmethod
    def get_char_id(word):
        if len(word) == 0:
            return None
        valid_indices = [id for id, char in enumerate(word) if char in eng_characters]
        if len(word) - 1 in valid_indices:
            valid_indices.remove(len(word) - 1)
        if len(valid_indices) == 0:
            return None
        char_id = np.random.choice(valid_indices, size=1)[0]
        return char_id

    def get_pert_words(self, word, char_id, return_all_possible=False):
        if isinstance(word, str):
            word = list(word)
        word = word.copy()

        if char_id >= len(word) - 1:
            return ["".join(word)]

        char1, char2 = word[char_id], word[char_id + 1]
        word[char_id], word[char_id + 1] = char2, char1
        pert_words = ["".join(word)]
        return pert_words


class SubstituteCharacterAttacker(BaseCharacterAttacker):
    def get_pert_words(self, word, char_id, return_all_possible=False):
        if isinstance(word, str):
            word = list(word)
        word = word.copy()

        if self.keyboard_constrain:
            target_char = word[char_id]
            if target_char in nearby_characters:
                candidate_chars = list(nearby_characters[target_char])
                if self.allow_numeric and target_char in nearby_numerics:
                    candidate_chars += list(nearby_numerics[target_char])

                if return_all_possible:
                    substitute_chars = candidate_chars
                else:
                    substitute_chars = np.random.choice(candidate_chars, size=1)
            else:
                substitute_chars = [target_char]
        else:
            candidate_chars = eng_characters
            if self.allow_numeric:
                candidate_chars += numeric_characters

            if return_all_possible:
                substitute_chars = candidate_chars
            else:
                substitute_chars = np.random.choice(candidate_chars, size=1)
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
    def __init__(self, insert_p=0.25, drop_p=0.25, swap_p=0.25, substitute_p=0.25, keyboard_constrain=False, allow_numeric=False, random_seed=None):
        self.insert_p = insert_p
        self.drop_p = drop_p
        self.swap_p = swap_p
        self.substitute_p = substitute_p
        self.insert_attacker = InsertCharacterAttacker(keyboard_constrain, allow_numeric, random_seed)
        self.drop_attacker = DropCharacterAttacker(keyboard_constrain, allow_numeric, random_seed)
        self.swap_attacker = SwapCharacterAttacker(keyboard_constrain, allow_numeric, random_seed)
        self.substitute_attacker = SubstituteCharacterAttacker(keyboard_constrain, allow_numeric, random_seed)

    def _augment(self,
            sentence=None, 
            tokens=None, 
            perturb_num=1, 
            min_char_perturb_num=1,
            max_char_perturb_num=1,
            min_word_len=3,
            word_indices=None, 
            attack_methods=None, 
            char_indices=None, 
            return_tokens=False
        ):
        """
        word_indices: a list of [word_id, ...]
        attack_methods: a list of [attack_method, ...] or [[attack_method, ...], ...]
        char_indices: a list of [char_id, ...] or [[char_id, ...], ...]
        """
        if sentence is not None:
            assert isinstance(sentence, str)
        if tokens is not None:
            assert isinstance(tokens, list)
        if sentence is not None and tokens is not None:
            print("sentence and tokens arguments are provided, ignore sentence argument.")
        
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
            valid_indices = [id for id in np.arange(len(tokens)) if word_validation(tokens[id], min_word_len)]
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
        # Get character indices
        if char_indices is None:
            # Determine number of character perturbations for each word_id
            char_perturb_nums = np.random.choice(list(range(min_char_perturb_num, max_char_perturb_num + 1)), size=len(word_indices), replace=True)
            char_indices = [[None] * char_perturb_num for char_perturb_num in char_perturb_nums]
        else:
            char_indices = [sub_char_indices if isinstance(sub_char_indices, list) else [sub_char_indices] for sub_char_indices in char_indices]    
        # Get attacking methods
        if attack_methods is None:
            attack_methods = [np.random.choice(["insert", "drop", "swap", "substitute"], size=char_perturb_num, replace=True, p=[self.insert_p, self.drop_p, self.swap_p, self.substitute_p]) for char_perturb_num in char_perturb_nums]
        else:
            attack_methods = [sub_attack_methods if isinstance(sub_attack_methods, list) else [sub_attack_methods] for sub_attack_methods in attack_methods]    
        # Attack
        pert_tokens = tokens.copy()
        for word_id, sub_attack_methods, sub_char_indices in zip(word_indices, attack_methods, char_indices):
            for attack_method, char_id in zip(sub_attack_methods, sub_char_indices):
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

    def __call__(self, *args, n=1, **kwargs):
        assert n >= 1, "n must be equal or more than 1"
        if n == 1:
            return self._augment(*args, **kwargs)
        else:
            return [self._augment(*args, **kwargs) for _ in range(n)]


class BaseAdversarialAttacker:
    @abstractmethod
    def encode(self, sentences):
        raise NotImplementedError("Please implement this method which receives list of sentences and returns tensor of size (sentence_num, embedding_dim)")

    def candidates_ranking(self, candidates, references):
        if not isinstance(references, list):
            references = [references]
        # Embedding
        reference_embeddings = self.encode(references)     # (reference_size, embedding_dim)
        candidate_embeddings = self.encode(candidates)     # (search_size, embedding_dim)
        # Calculate Similarity
        similarity = torch.nn.CosineSimilarity(dim=-1)(reference_embeddings.unsqueeze(1), candidate_embeddings.unsqueeze(0))  # (reference_size, search_size)
        similarity, _ = torch.max(similarity, dim=0)  # (search_size, )
        # Ranking
        sorted_indices = torch.argsort(similarity, dim=-1, descending=False)
        return sorted_indices

    def word_importance_ranking(self, words, references=None):
        if len(words) == 1:
            return [0]
        # Get candidates
        candidates = [words2sentence([word for i, word in enumerate(words) if i != word_id]) for word_id in range(len(words))]
        # Ranking
        if references is None:
            references = [words2sentence(words)]
        sorted_indices = self.candidates_ranking(candidates, references)
        return sorted_indices

    def char_importance_ranking(self, words, target_word_id, references=None):
        if len(words[target_word_id]) == 1:
            return [0]
        # Get candidates
        candidates = []
        chars = word2chars(words[target_word_id])
        for char_id in range(len(chars)):
            pert_word = chars2word(chars[:char_id] + chars[char_id+1:])
            pert_sentence = words2sentence([words[word_id] if word_id != target_word_id else pert_word for word_id in range(len(words))])
            candidates.append(pert_sentence)
        # Ranking
        if references is None:
            references = [words2sentence(words)]
        sorted_indices = self.candidates_ranking(candidates, references)
        return sorted_indices


class AdversarialFilter(BaseAdversarialAttacker):
    def __call__(self, candidates, references, top_k=1, last_k=1):
        sorted_indices = self.candidates_ranking(candidates, references)

        returns = {}
        # Get top k
        if top_k > 0:
            top_k_sentences = [candidates[sorted_indices[i]] for i in range(top_k)]
            returns["top_k"] = top_k_sentences
        # Get last k
        if last_k > 0:
            last_k_sentences = [candidates[sorted_indices[-(i+1)]] for i in range(last_k)]
            returns["last_k"] = last_k_sentences
        return returns


class ClusteringFilter(BaseAdversarialAttacker):
    def __call__(self, candidates, cluster_num=5):
        # Embeddings
        embeddings = self.encode(candidates).cpu().detach().numpy()
        # Clustering
        cluster_indices = {cluster_id: [] for cluster_id in range(cluster_num)}
        clusters = KMeans(cluster_num, random_state=0).fit(embeddings)
        for embedding_id, cluster_id in enumerate(clusters.labels_):
            cluster_indices[cluster_id].append(embedding_id)
        # Sampling sentences from each cluster
        p = {cluster_id: clusters.transform(embeddings[embedding_indices])[:, cluster_id] for cluster_id, embedding_indices in cluster_indices.items()}
        p = {cluster_id: distances / np.sum(distances) if np.sum(distances) > 0 else np.ones_like(distances) for cluster_id, distances in p.items()}
        candidate_indices = [np.random.choice(cluster_indices[cluster_id], size=1, p=p[cluster_id])[0] for cluster_id in range(cluster_num)]
        sentences = [candidates[candidate_id] for candidate_id in candidate_indices]
        return sentences


class ClusteringRandomCharacterAttacker:
    def __init__(self, filter, search_size=10, **kwargs):
        self.filter = filter
        self.search_size = search_size
        self.attacker = RandomCharacterAttacker(**kwargs)

    def get_candidates(self, sentence, perturb_num=1, min_char_perturb_num=1, max_char_perturb_num=1, min_word_len=3):
        candidates = self.attacker(sentence, 
                                   n=self.search_size, 
                                   perturb_num=perturb_num, 
                                   min_char_perturb_num=min_char_perturb_num, 
                                   max_char_perturb_num=max_char_perturb_num,
                                   min_word_len=min_word_len)
        return list(set(candidates))

    def __call__(self, 
                 sentence, 
                 cluster_num=5, 
                 perturb_num=1, 
                 min_char_perturb_num=1,
                 max_char_perturb_num=1,
                 min_word_len=3,
                ):
        # Generate candidates
        candidates = self.get_candidates(sentence, perturb_num, min_char_perturb_num, max_char_perturb_num, min_word_len)
        return self.filter(candidates, cluster_num=cluster_num)
        

class RandomSearchAdversarialCharacterAttacker:
    def __init__(self, filter, search_size=10, **kwargs):
        self.filter = filter
        self.search_size = search_size
        self.attacker = RandomCharacterAttacker(**kwargs)

    def get_candidates(self, sentence, perturb_num=1, min_char_perturb_num=1, max_char_perturb_num=1, min_word_len=3):
        candidates = self.attacker(sentence, 
                                   n=self.search_size, 
                                   perturb_num=perturb_num, 
                                   min_char_perturb_num=min_char_perturb_num, 
                                   max_char_perturb_num=max_char_perturb_num,
                                   min_word_len=min_word_len)
        return list(set(candidates))

    def __call__(self, 
                 sentence, 
                 top_k=1, 
                 last_k=1,
                 perturb_num=1, 
                 min_char_perturb_num=1,
                 max_char_perturb_num=1,
                 min_word_len=3,
                ):
        # Generate candidates
        candidates = self.get_candidates(sentence, perturb_num, min_char_perturb_num, max_char_perturb_num, min_word_len)
        return self.filter(candidates, sentence, top_k=top_k, last_k=last_k)


class DiversionRandomSearchAdversarialCharacterAttacker(RandomSearchAdversarialCharacterAttacker):
    def __call__(self, 
                 sentence, 
                 top_k=1, 
                 last_k=1,
                 perturb_num=1, 
                 min_char_perturb_num=1,
                 max_char_perturb_num=1,
                 min_word_len=3,
                ):
        returns = {}
        # Top k
        if top_k > 0:
            top_k_sentences = []
            references = [sentence]
            for _ in range(top_k):
                # Generate candidates
                candidates = self.get_candidates(sentence, perturb_num, min_char_perturb_num, max_char_perturb_num, min_word_len)
                fillered_candidate = self.filter(candidates, references, top_k=1, last_k=0)["top_k"][0]
                top_k_sentences.append(fillered_candidate)
                references.append(fillered_candidate)
            returns["top_k"] = top_k_sentences
        # Last k
        if last_k > 0:
            last_k_sentences = []
            references = [sentence]
            for _ in range(last_k):
                # Generate candidates
                candidates = self.get_candidates(sentence, perturb_num, min_char_perturb_num, max_char_perturb_num, min_word_len)
                fillered_candidate = self.filter(candidates, references, top_k=0, last_k=1)["last_k"][0]
                last_k_sentences.append(fillered_candidate)
                references.append(fillered_candidate)
            returns["last_k"] = last_k_sentences
        return returns


class NarrowSearchAdversarialCharacterAttacker(BaseAdversarialAttacker):
    def __init__(self, *args, search_size=10, **kwargs):
        self.attacker = RandomCharacterAttacker(*args, **kwargs)
        self.search_size = search_size

    def get_candidates(self, tokens, word_id, char_perturb_num=1, min_word_len=3):
        candidates = self.attacker(tokens=tokens, 
                                   n=self.search_size, 
                                   min_char_perturb_num=char_perturb_num, 
                                   max_char_perturb_num=char_perturb_num, 
                                   min_word_len=min_word_len, 
                                   word_indices=[word_id])
        return list(set(candidates))

    def generate_adversarial_samples(self, 
            original_sent, 
            k=1,
            mode="top_k", 
            perturb_num=1, 
            min_char_perturb_num=1, 
            max_char_perturb_num=1, 
            min_word_len=3
        ):
        # Get word indices sorted by importantness
        words = sentence2words(original_sent)
        valid_indices = [word_id for word_id in np.arange(len(words)) if word_validation(words[word_id], min_word_len)]
        sorted_word_indices = [word_id for word_id in self.word_importance_ranking(words) if word_id in valid_indices]

        if len(valid_indices) == 0:
            return [original_sent]

        if isinstance(perturb_num, int):
            perturb_num = min(perturb_num, len(valid_indices))
        else:
            perturb_num = math.ceil(perturb_num * len(valid_indices))

        # Determine number of character perturbations for each word_id
        char_perturb_nums = np.random.choice(list(range(min_char_perturb_num, max_char_perturb_num + 1)), size=perturb_num, replace=True)

        adv_samples = []
        pert_words = words.copy()
        for i in range(perturb_num):
            if mode == "top_k" or mode == "maxmin_k":
                target_word_id = sorted_word_indices[i]
            else:
                target_word_id = sorted_word_indices[-(i+1)]
            char_perturb_num = char_perturb_nums[i]
            # Generate candidates
            candidates = self.get_candidates(pert_words, target_word_id, char_perturb_num, min_word_len)
            sorted_indices = self.candidates_ranking(candidates, original_sent)
            # Update pert_words
            if i < perturb_num - 1:
                if mode == "top_k" or mode == "minmax_k":
                    pert_words = sentence2words(candidates[sorted_indices[0]])
                else:
                    pert_words = sentence2words(candidates[sorted_indices[-1]])
            else:
                for k in range(min(k, len(candidates))):
                    t_pert_words = pert_words.copy()
                    if mode == "top_k" or mode == "minmax_k":
                        t_pert_words = sentence2words(candidates[sorted_indices[k]])
                    else:
                        t_pert_words = sentence2words(candidates[sorted_indices[-(k+1)]])
                    adv_samples.append(words2sentence(t_pert_words))
                if len(adv_samples) < k:
                    print(f"(Warning): {mode} of {k} is too large. Output size of {mode} will be smaller than expected.")
        return adv_samples

    def __call__(self,
            sentence,
            top_k=1,
            last_k=1,
            minmax_k=1,
            maxmin_k=1,
            perturb_num=1,
            min_char_perturb_num=1,
            max_char_perturb_num=1,
            min_word_len=3,
        ):
        returns = {}
        # Top k
        if top_k > 0:
            top_k_sents = self.generate_adversarial_samples(sentence, 
                                                            k=top_k, 
                                                            mode="top_k", 
                                                            perturb_num=perturb_num, 
                                                            min_char_perturb_num=min_char_perturb_num, 
                                                            max_char_perturb_num=max_char_perturb_num, 
                                                            min_word_len=min_word_len)
            returns["top_k"] = top_k_sents
        # Last k
        if last_k > 0:
            last_k_sents = self.generate_adversarial_samples(sentence, 
                                                             k=last_k, 
                                                             mode="last_k", 
                                                             perturb_num=perturb_num, 
                                                             min_char_perturb_num=min_char_perturb_num, 
                                                             max_char_perturb_num=max_char_perturb_num, 
                                                             min_word_len=min_word_len)
            returns["last_k"] = last_k_sents
        # Min-Max k
        if minmax_k > 0:
            minmax_k_sents = self.generate_adversarial_samples(sentence, 
                                                             k=minmax_k, 
                                                             mode="minmax_k", 
                                                             perturb_num=perturb_num, 
                                                             min_char_perturb_num=min_char_perturb_num, 
                                                             max_char_perturb_num=max_char_perturb_num, 
                                                             min_word_len=min_word_len)
            returns["minmax_k"] = minmax_k_sents
        # Max-Min k
        if maxmin_k > 0:
            maxmin_k_sents = self.generate_adversarial_samples(sentence, 
                                                             k=maxmin_k, 
                                                             mode="maxmin_k", 
                                                             perturb_num=perturb_num, 
                                                             min_char_perturb_num=min_char_perturb_num, 
                                                             max_char_perturb_num=max_char_perturb_num, 
                                                             min_word_len=min_word_len)
            returns["maxmin_k"] = maxmin_k_sents
        return returns


class DiversionNarrowSearchAdversarialCharacterAttacker(NarrowSearchAdversarialCharacterAttacker):
    def generate_adversarial_samples(self, 
            original_sent, 
            k=1,
            mode="top_k", 
            perturb_num=1, 
            min_char_perturb_num=1, 
            max_char_perturb_num=1, 
            min_word_len=3,
        ):
        words = sentence2words(original_sent)
        valid_indices = [word_id for word_id in np.arange(len(words)) if word_validation(words[word_id], min_word_len)]

        if len(valid_indices) == 0:
            return [original_sent]

        if isinstance(perturb_num, int):
            perturb_num = min(perturb_num, len(valid_indices))
        else:
            perturb_num = math.ceil(perturb_num * len(valid_indices))

        adv_samples = []
        references = [original_sent]
        for _ in range(k):
            pert_words = words.copy()
            # Get word indices sorted by importantness
            sorted_word_indices = [word_id for word_id in self.word_importance_ranking(words, references) if word_id in valid_indices]
            # Determine number of character perturbations for each word_id
            char_perturb_nums = np.random.choice(list(range(min_char_perturb_num, max_char_perturb_num + 1)), size=perturb_num, replace=True)
            for i in range(perturb_num):
                if mode == "top_k" or mode == "maxmin_k":
                    target_word_id = sorted_word_indices[i]
                else:
                    target_word_id = sorted_word_indices[-(i+1)]
                char_perturb_num = char_perturb_nums[i]
                # Generate candidates
                candidates = self.get_candidates(pert_words, target_word_id, char_perturb_num, min_word_len)
                sorted_indices = self.candidates_ranking(candidates, references)
                # Update pert_words
                if mode == "top_k" or mode == "minmax_k":
                    pert_words = sentence2words(candidates[sorted_indices[0]])
                else:
                    pert_words = sentence2words(candidates[sorted_indices[-1]])
                # Get adv_sample
                if i >= perturb_num - 1:
                    adv_sample = words2sentence(pert_words)
                    adv_samples.append(adv_sample)
                    references.append(adv_sample)
        return adv_samples


class BeamSearchAdversarialCharacterAttacker(BaseAdversarialAttacker):
    def __init__(self, allow_insert=True, allow_drop=True, allow_swap=True, allow_substitute=True, keyboard_constrain=False, allow_numeric=False):
        self.allow_insert = allow_insert
        self.allow_drop = allow_drop
        self.allow_swap = allow_swap
        self.allow_substitute = allow_substitute
        self.insert_attacker = InsertCharacterAttacker(keyboard_constrain, allow_numeric)
        self.drop_attacker = DropCharacterAttacker(keyboard_constrain, allow_numeric)
        self.swap_attacker = SwapCharacterAttacker(keyboard_constrain, allow_numeric)
        self.substitute_attacker = SubstituteCharacterAttacker(keyboard_constrain, allow_numeric)

    def get_candidates(self, tokens, word_id, char_id):
        target_word = list(tokens[word_id])

        perturb_words = []
        if self.allow_insert:
            perturb_words.extend(self.insert_attacker.get_pert_words(target_word, char_id, return_all_possible=True))
        if self.allow_drop:
            perturb_words.extend(self.drop_attacker.get_pert_words(target_word, char_id, return_all_possible=True))
        if self.allow_swap:
            perturb_words.extend(self.swap_attacker.get_pert_words(target_word, char_id, return_all_possible=True))
        if self.allow_substitute:
            perturb_words.extend(self.substitute_attacker.get_pert_words(target_word, char_id, return_all_possible=True))
        
        candidates = []
        for perturb_word in perturb_words:
            t_words = tokens.copy()
            t_words[word_id] = perturb_word
            perturb_sentence = words2sentence(t_words)
            candidates.append(perturb_sentence)
        return candidates

    def generate_adversarial_samples(self, 
            original_sent, 
            k=1,
            mode="top_k", 
            perturb_num=1, 
            min_char_perturb_num=1, 
            max_char_perturb_num=1, 
            min_word_len=3
        ):
        # Get word indices sorted by importantness
        words = sentence2words(original_sent)
        valid_indices = [word_id for word_id in np.arange(len(words)) if word_validation(words[word_id], min_word_len)]
        sorted_word_indices = [word_id for word_id in self.word_importance_ranking(words) if word_id in valid_indices]

        if len(valid_indices) == 0:
            return [original_sent]

        if isinstance(perturb_num, int):
            perturb_num = min(perturb_num, len(valid_indices))
        else:
            perturb_num = math.ceil(perturb_num * len(valid_indices))

        # Determine number of character perturbations for each word_id
        char_perturb_nums = np.random.choice(list(range(min_char_perturb_num, max_char_perturb_num + 1)), size=perturb_num, replace=True)

        adv_samples = []
        pert_words = words.copy()
        for i in range(perturb_num):
            if mode == "top_k" or mode == "maxmin_k" or mode == "maxminmax_k":
                target_word_id = sorted_word_indices[i]
            else:
                target_word_id = sorted_word_indices[-(i+1)]
            for j in range(min(len(pert_words[target_word_id]), char_perturb_nums[i])):
                # Get character indices sorted by importantness
                sorted_char_indices = self.char_importance_ranking(pert_words, target_word_id)
                if mode == "top_k" or mode == "minmax_k" or mode == "minmaxmin_k":
                    target_char_id = sorted_char_indices[0]
                else:
                    target_char_id = sorted_char_indices[-1]
                # Generate candidates
                candidates = self.get_candidates(pert_words, target_word_id, target_char_id)
                sorted_indices = self.candidates_ranking(candidates, original_sent)
                # Update pert_words
                if i < perturb_num - 1 or j < char_perturb_nums[i] - 1:
                    if mode == "top_k" or mode == "minmax_k" or mode == "maxminmax_k":
                        pert_words = sentence2words(candidates[sorted_indices[0]])
                    else:
                        pert_words = sentence2words(candidates[sorted_indices[-1]])
                else:
                    for k in range(min(k, len(candidates))):
                        t_pert_words = pert_words.copy()
                        if mode == "top_k" or mode == "minmax_k" or mode == "maxminmax_k":
                            t_pert_words = sentence2words(candidates[sorted_indices[k]])
                        else:
                            t_pert_words = sentence2words(candidates[sorted_indices[-(k+1)]])
                        adv_samples.append(words2sentence(t_pert_words))
                    if len(adv_samples) < k:
                        print(f"(Warning): {mode} of {k} is too large. Output size of {mode} will be smaller than expected.")
        return adv_samples

    def __call__(self, 
            sentence, 
            top_k=1, 
            last_k=1,
            minmax_k=1,
            maxmin_k=1,
            minmaxmin_k=1,
            maxminmax_k=1,
            perturb_num=1, 
            min_char_perturb_num=1, 
            max_char_perturb_num=1, 
            min_word_len=3
        ):
        returns = {}
        # Top k
        if top_k > 0:
            top_k_sents = self.generate_adversarial_samples(sentence, 
                                                            k=top_k, 
                                                            mode="top_k", 
                                                            perturb_num=perturb_num, 
                                                            min_char_perturb_num=min_char_perturb_num, 
                                                            max_char_perturb_num=max_char_perturb_num, 
                                                            min_word_len=min_word_len)
            returns["top_k"] = top_k_sents
        # Last k
        if last_k > 0:
            last_k_sents = self.generate_adversarial_samples(sentence, 
                                                            k=last_k, 
                                                            mode="last_k", 
                                                            perturb_num=perturb_num, 
                                                            min_char_perturb_num=min_char_perturb_num, 
                                                            max_char_perturb_num=max_char_perturb_num, 
                                                            min_word_len=min_word_len)
            returns["last_k"] = last_k_sents
        # Min-Max k
        if minmax_k > 0:
            minmax_k_sents = self.generate_adversarial_samples(sentence, 
                                                            k=minmax_k, 
                                                            mode="minmax_k", 
                                                            perturb_num=perturb_num, 
                                                            min_char_perturb_num=min_char_perturb_num, 
                                                            max_char_perturb_num=max_char_perturb_num, 
                                                            min_word_len=min_word_len)
            returns["minmax_k"] = minmax_k_sents
        # Max-Min k
        if maxmin_k > 0:
            maxmin_k_sents = self.generate_adversarial_samples(sentence, 
                                                            k=maxmin_k, 
                                                            mode="maxmin_k", 
                                                            perturb_num=perturb_num, 
                                                            min_char_perturb_num=min_char_perturb_num, 
                                                            max_char_perturb_num=max_char_perturb_num, 
                                                            min_word_len=min_word_len)
            returns["maxmin_k"] = maxmin_k_sents
        # Min-Max-Min k
        if minmaxmin_k > 0:
            minmaxmin_k_sents = self.generate_adversarial_samples(sentence, 
                                                            k=minmaxmin_k, 
                                                            mode="minmaxmin_k", 
                                                            perturb_num=perturb_num, 
                                                            min_char_perturb_num=min_char_perturb_num, 
                                                            max_char_perturb_num=max_char_perturb_num, 
                                                            min_word_len=min_word_len)
            returns["minmaxmin_k"] = minmaxmin_k_sents
        # Max-Min-Max k
        if maxminmax_k > 0:
            maxminmax_k_sents = self.generate_adversarial_samples(sentence, 
                                                            k=maxminmax_k, 
                                                            mode="maxminmax_k", 
                                                            perturb_num=perturb_num, 
                                                            min_char_perturb_num=min_char_perturb_num, 
                                                            max_char_perturb_num=max_char_perturb_num, 
                                                            min_word_len=min_word_len)
            returns["maxminmax_k"] = maxminmax_k_sents
        return returns


class DiversionBeamSearchAdversarialCharacterAttacker(BeamSearchAdversarialCharacterAttacker):
    def generate_adversarial_samples(self, 
            original_sent, 
            k=1,
            mode="top_k", 
            perturb_num=1, 
            min_char_perturb_num=1, 
            max_char_perturb_num=1, 
            min_word_len=3
        ):
        words = sentence2words(original_sent)
        valid_indices = [word_id for word_id in np.arange(len(words)) if word_validation(words[word_id], min_word_len)]

        if len(valid_indices) == 0:
            return [original_sent]

        if isinstance(perturb_num, int):
            perturb_num = min(perturb_num, len(valid_indices))
        else:
            perturb_num = math.ceil(perturb_num * len(valid_indices))

        adv_samples = []
        references = [original_sent]
        for _ in range(k):
            pert_words = words.copy()
            # Get word indices sorted by importantness
            sorted_word_indices = [word_id for word_id in self.word_importance_ranking(words, references) if word_id in valid_indices]
            # Determine number of character perturbations for each word_id
            char_perturb_nums = np.random.choice(list(range(min_char_perturb_num, max_char_perturb_num + 1)), size=perturb_num, replace=True)
            for i in range(perturb_num):
                if mode == "top_k" or mode == "maxmin_k" or mode == "maxminmax_k":
                    target_word_id = sorted_word_indices[i]
                else:
                    target_word_id = sorted_word_indices[-(i+1)]
                for j in range(min(len(pert_words[target_word_id]), char_perturb_nums[i])):
                    # Get character indices sorted by importantness
                    sorted_char_indices = self.char_importance_ranking(pert_words, target_word_id, references)
                    if mode == "top_k" or mode == "minmax_k" or mode == "minmaxmin_k":
                        target_char_id = sorted_char_indices[0]
                    else:
                        target_char_id = sorted_char_indices[-1]
                    # Generate candidates
                    candidates = self.get_candidates(pert_words, target_word_id, target_char_id)
                    sorted_indices = self.candidates_ranking(candidates, references)
                    # Update pert_words
                    if mode == "top_k" or mode == "minmax_k" or mode == "maxminmax_k":
                        pert_words = sentence2words(candidates[sorted_indices[0]])
                    else:
                        pert_words = sentence2words(candidates[sorted_indices[-1]])
                    # Get adv_sample
                    if i >= perturb_num - 1 and j >= char_perturb_nums[i] - 1:
                        adv_sample = words2sentence(pert_words)
                        adv_samples.append(adv_sample)
                        references.append(adv_sample)
        return adv_samples