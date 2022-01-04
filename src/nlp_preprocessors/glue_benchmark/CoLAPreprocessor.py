from .BasePreprocessor import BasePreprocessor


class CoLAPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(sent_num=1, **kwargs)