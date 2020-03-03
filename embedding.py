import numpy as np


class pretrained_embedding:
    def __init__(self, embedding_path):
        self.word_to_index = {}
        self.word_vecs = []
        self.load_embedding(embedding_path)
        self.dimension = 100

    def load_embedding(self, path):
        with open(path) as fd:
            for line in fd.readlines():
                line = line.split()
                word = line[0]
                vect = list(map(float, line[1:]))
                self.word_to_index[word] = len(self.word_to_index)
                self.word_vecs.append(np.array(vect))

    def __getitem__(self, item):
        if type(item) == int:
            if item == -1:
                return np.zeros(self.dimension)
            else:
                return self.word_vecs[item]

        if item not in self.word_to_index:
            return None

        return self.word_vecs[self.word_to_index[item]]
