from .BasePreprocessor import BasePreprocessor


class MRPCPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(sent_num=2, **kwargs)