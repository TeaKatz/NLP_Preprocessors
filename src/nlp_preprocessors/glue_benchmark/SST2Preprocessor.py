from .BasePreprocessor import BasePreprocessor


class SST2Preprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(sent_num=1, **kwargs)