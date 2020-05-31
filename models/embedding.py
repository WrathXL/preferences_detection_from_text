import numpy as np
import os

class pretrained_embedding:
    def __init__(self, embedding_path):
        self.word_to_index = {}
        self.word_vecs = []
        self.load_embedding(embedding_path)

        self.dimension = 100
        self.vocabulary_size = len(self.word_to_index)

    def load_embedding(self, path):
        self.word_vecs.append(np.zeros((100)))
        self.word_to_index['PAD'] = 0

        with open(path) as fd:
            for line in fd.readlines():
                line = line.split()
                word = line[0]
                vect = list(map(float, line[1:]))
                self.word_to_index[word] = len(self.word_to_index)
                self.word_vecs.append(np.array(vect))
        self.word_vecs = np.array(self.word_vecs)

    def __getitem__(self, item):
        if type(item) == int:
            if item == -1:
                return np.zeros(self.dimension)
            else:
                return self.word_vecs[item]
        if item not in self.word_to_index:
            return None
        return self.word_vecs[self.word_to_index[item]]


if __name__ == '__main__':
    e = pretrained_embedding('data/glove.6B.100d.txt')