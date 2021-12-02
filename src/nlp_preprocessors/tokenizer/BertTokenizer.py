from typing import List, Dict, Union
from transformers import BertTokenizer

from ..padding import AutoTokenIdPadding


class SubwordBertTokenizer(BertTokenizer):
    def __call__(self, 
                texts: Union[List[str], str], 
                padding: bool=False,
                padding_length: Union[str, int]="longest",
                max_padding_length: int=None,
                padding_id: int=0,
                return_tensors: Union[str, bool]=False,):

        features = super().__call__(texts)
        if isinstance(texts, list):
            if padding:
                features = self.pad_features(features, 
                                            padding_length=padding_length,
                                            max_padding_length=max_padding_length,
                                            padding_id=padding_id,
                                            return_tensors=return_tensors)
        return features

    def pad_features(self, 
            features: Dict[str, List],
            padding_length: Union[str, int]="longest",
            max_padding_length: int=None,
            padding_id: int=0,
            return_tensors: Union[str, bool]=False,
            **kwargs):
        
        padding = AutoTokenIdPadding(padding_length=padding_length, 
                                    max_padding_length=max_padding_length,
                                    padding_id=padding_id)

        features["input_ids"], features["attention_mask"] = padding(features["input_ids"], return_padding_masks=True, return_tensors=return_tensors)
        if "token_type_ids" in features:
            features["token_type_ids"] = padding(features["token_type_ids"], return_tensors=return_tensors)
        return features


class FullwordBertTokenizer(BertTokenizer):
    def process_text(self, text):
        dict = super().__call__(text)

        tokens = self.tokenize(text)
        features = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        for i, token in enumerate(tokens):
            if "##" in token:
                features["input_ids"][-1].append(dict["input_ids"][i + 1])
            else:
                features["input_ids"].append([dict["input_ids"][i + 1]])
                features["token_type_ids"].append(dict["token_type_ids"][i + 1])
                features["attention_mask"].append(dict["attention_mask"][i + 1])

        # Add CLS and SEP
        features["input_ids"] = [[dict["input_ids"][0]]] + features["input_ids"] + [[dict["input_ids"][-1]]]
        features["token_type_ids"] = [dict["token_type_ids"][0]] + features["token_type_ids"] + [dict["token_type_ids"][-1]]
        features["attention_mask"] = [dict["attention_mask"][0]] + features["attention_mask"] + [dict["attention_mask"][-1]]
        return features

    def process_texts(self, texts):
        features = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        for text in texts:
            _features = self.process_text(text)
            features["input_ids"].append(_features["input_ids"])
            features["token_type_ids"].append(_features["token_type_ids"])
            features["attention_mask"].append(_features["attention_mask"])
        return features

    def __call__(self, 
                texts: List[str], 
                padding: bool=False,
                padding_length: Union[str, int]="longest",
                sub_padding_length: Union[str, int]="longest",
                max_padding_length: int=None,
                max_sub_padding_length: int=None,
                padding_id: int=0,
                return_tensors: Union[str, bool]=False):

        if isinstance(texts, str):
            features = self.process_text(texts)
        else:
            features = self.process_texts(texts)
            if padding:
                features = self.pad_features(features, 
                                            padding_length=padding_length,
                                            sub_padding_length=sub_padding_length,
                                            max_padding_length=max_padding_length,
                                            max_sub_padding_length=max_sub_padding_length,
                                            padding_id=padding_id,
                                            return_tensors=return_tensors)
        return features

    def pad_features(self, 
            features: Dict[str, List],
            padding_length: Union[str, int]="longest",
            sub_padding_length: Union[str, int]="longest",
            max_padding_length: int=None,
            max_sub_padding_length: int=None,
            padding_id: int=0,
            return_tensors: Union[str, bool]=False,
            **kwargs):
        
        padding = AutoTokenIdPadding(padding_length=padding_length, 
                                    sub_padding_length=sub_padding_length,
                                    max_padding_length=max_padding_length,
                                    max_sub_padding_length=max_sub_padding_length,
                                    padding_id=padding_id)

        features["input_ids"], features["attention_mask"], features["sub_attention_mask"] = padding(features["input_ids"], 
                                                                                                   return_padding_masks=True, 
                                                                                                   return_sub_padding_masks=True, 
                                                                                                   return_tensors=return_tensors)
        if "token_type_ids" in features:
            features["token_type_ids"] = padding(features["token_type_ids"], return_tensors=return_tensors)
        return features