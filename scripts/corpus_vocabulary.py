import os, json
from corpus_process import Corpus

files = ['sensitive1.tsv', 'sensitive2.tsv', 'sensitive3.tsv']
tsv_path = [os.path.join(os.pardir, 'data', f) for f in files]
vocabulary_path = os.path.join(os.pardir, 'data', 'vocabulary_corpus.json')

corpus = Corpus(tsv_path)

x, _ = corpus.read_tsv()

vocabulary = {}
for sent in x:
    for tok in sent:
        if not tok in vocabulary:
            vocabulary[tok] = len(vocabulary) + 1


with open(vocabulary_path, 'w') as fd:
    json.dump(vocabulary, fd)