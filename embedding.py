import numpy as np

class pretrained_embedding:
    def __init__(self):
        self.word_to_index = {}
        self.word_vecs = []
        self.load_embedding('data/glove.6B.100d.txt')

    def load_embedding(self, path):
        with open(path) as fd:
            for line in fd.readlines():
                line = line.split()
                word = line[0]
                vect = list(map(float, line[1:]))
                self.word_to_index[word] = len(self.word_to_index)
                self.word_vecs.append(np.array(vect))


    def __getitem__(self, item):
        if item not in self.word_to_index: return None
        return self.word_vecs[self.word_to_index[item]]