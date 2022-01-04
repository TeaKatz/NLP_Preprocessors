from .BasePreprocessor import BasePreprocessor


class MNLIPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(sent_num=2, **kwargs)