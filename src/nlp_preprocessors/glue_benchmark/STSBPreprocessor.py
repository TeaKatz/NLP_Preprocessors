from .BasePreprocessor import BasePreprocessor


class STSBPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(sent_num=2, **kwargs)