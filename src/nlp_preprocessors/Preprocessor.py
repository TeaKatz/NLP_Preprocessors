import os
import numpy as np


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


class ProcessShortcut:
    def __init__(self, default_process: callable, local_dir: str="shortcut"):
        self.default_process = default_process
        self.local_dir = local_dir
        self.data = {}

        self.load_local()

    def save_local(self):
        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)
        np.savez(self.local_dir + "/data.npz", **self.data)

    def load_local(self):
        if os.path.exists(self.local_dir + "/data.npz"):
            npzfile = np.load(self.local_dir + "/data.npz")
            self.data = {key: npzfile[key] for key in npzfile.files}

    def __call__(self, *inputs, key=None):
        if key is None:
            key = inputs

        if key in self.data:
            outputs = self.data[key]
        else:
            outputs = self.default_process(*inputs)

        if key not in self.data:
            self.data[key] = outputs
            self.save_local()
        return outputs
