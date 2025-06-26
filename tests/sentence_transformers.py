class SentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, convert_to_numpy=True):
        import numpy as np
        return np.array([len(text)], dtype=float)

class util:
    @staticmethod
    def cos_sim(a, b):
        return [[0.0]]
