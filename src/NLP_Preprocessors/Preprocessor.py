class SequentialPreprocessor:
    def __init__(self, preprocessors):
        if not isinstance(preprocessors, list):
            preprocessors = [preprocessors]

        self.preprocessors = preprocessors

    def __call__(self, *args, **kwargs):
        outputs = self.preprocessors[0](*args, **kwargs)
        for i in range(1, len(self.preprocessors)):
            outputs = self.preprocessors[i](outputs)
        return outputs
