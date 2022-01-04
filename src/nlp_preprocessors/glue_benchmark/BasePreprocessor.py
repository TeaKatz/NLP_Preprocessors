from transformers import AutoTokenizer


class BasePreprocessor:
    def __init__(self, sent_num=1, test_set=False, padding_length="longest", max_padding_length=100):
        assert sent_num not in [1, 2]

        self.sent_num = sent_num
        self.test_set = test_set
        self.padding_length = padding_length
        self.max_padding_length = max_padding_length
        self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    def process_sample(self, sample):
        if self.sent_num == 1:
            sent = sample["sentence"].strip()
            sent = [sent]
        else:
            sent_1 = sample["sentence1"].strip()
            sent_2 = sample["sentence2"].strip()
            sent_1 = [sent_1]
            sent_2 = [sent_2]
            sent = sent_1 + sent_2

        if self.test_set:
            label = sample["index"]
        else:
            label = sample["label"]
        return sent, label

    def __call__(self, batch):
        batch_size = len(batch)
        # Extract features
        sents = []
        labels = []
        for sample in batch:
            sent, label = self.process_sample(sample)
            sents.extend(sent)
            labels.append(label)
        features = self.tokenizer(sents, padding=True, padding_length=self.padding_length, max_padding_length=self.max_padding_length, return_tensors="pt")
        # Unflatten
        if self.sent_num == 1:
            features = {key: item.view(batch_size, 1, -1) for key, item in features.items()}
        else:
            features = {key: item.view(batch_size, 2, -1) for key, item in features.items()}
        # Construct samples
        samples = {
            "input": {
                "input_ids": features["input_ids"],
                "attention_mask": features["attention_mask"]
            },
            "target": labels
        }
        return samples