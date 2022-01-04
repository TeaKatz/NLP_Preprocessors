from .BasePreprocessor import BasePreprocessor


class QNLIPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(sent_num=2, **kwargs)