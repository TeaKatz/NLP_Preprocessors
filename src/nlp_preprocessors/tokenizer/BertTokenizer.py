from transformers import BertTokenizer


class ModifiedBertTokenizer(BertTokenizer):
    def process_text(self, text):
        dict = super().__call__(text)

        tokens = self.tokenize(text)
        outputs = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        for i, token in enumerate(tokens):
            if "##" in token:
                outputs["input_ids"][-1].append(dict["input_ids"][i + 1])
            else:
                outputs["input_ids"].append([dict["input_ids"][i + 1]])
                outputs["token_type_ids"].append(dict["token_type_ids"][i + 1])
                outputs["attention_mask"].append(dict["attention_mask"][i + 1])

        # Add CLS and SEP
        outputs["input_ids"] = [[dict["input_ids"][0]]] + outputs["input_ids"] + [[dict["input_ids"][-1]]]
        outputs["token_type_ids"] = [dict["token_type_ids"][0]] + outputs["token_type_ids"] + [dict["token_type_ids"][-1]]
        outputs["attention_mask"] = [dict["attention_mask"][0]] + outputs["attention_mask"] + [dict["attention_mask"][-1]]
        return outputs

    def __call__(self, texts):
        if isinstance(texts, str):
            return self.process_text(texts)
        else:
            outputs = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
            for text in texts:
                _outputs = self.process_text(text)
                outputs["input_ids"].append(_outputs["input_ids"])
                outputs["token_type_ids"].append(_outputs["token_type_ids"])
                outputs["attention_mask"].append(_outputs["attention_mask"])
        return outputs