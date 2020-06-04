import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


from models.embedding import pretrained_embedding
from corpus_process import Corpus

files_paths = ['data/sensitive1.tsv', 'data/sensitive2.tsv', 'data/sensitive3.tsv']
vocabulary_path = 'data/vocabulary_glove.json'
embed_path = 'data/glove.6B.100d.txt'

emb = pretrained_embedding(embed_path)
corpus = Corpus(files_paths, vocabulary_path)

data = corpus.get_dataset()
X, Y = data.tensors

x = np.zeros((len(X), 74, 100))

for i in range(len(X)):
    for j in range(74):
        x[i, j] = emb[X[i, j].item()]