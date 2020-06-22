import nltk
import sklearn_crfsuite
import random
from corpus_process import Corpus
from sklearn_crfsuite import metrics
from seqeval.metrics import classification_report




def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]



files_paths = ['data/sensitive1.tsv', 'data/sensitive2.tsv', 'data/sensitive3.tsv']
corpus = Corpus(files_paths)
sents, tags = corpus.read_tsv()

data = []

for i, sent in enumerate(sents):
    tmp = []
    for j, t in enumerate(nltk.pos_tag(sent)):
        tmp.append((t[0], t[1], tags[i][j]))
    data.append(tmp)

random.shuffle(data)

X_train = [sent2features(s) for s in data[996:]]
y_train = [sent2labels(s) for s in data[996:]]

X_test = [sent2features(s) for s in data[:996]]
y_test = [sent2labels(s) for s in data[:996]]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)

if __name__ == '__main__':
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    y_p, y_t = [], []
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            y_p.append(y_pred[i][j])
            y_t.append(y_test[i][j])

    print(metrics.flat_classification_report(y_test, y_pred, labels=corpus.labels))
    print(classification_report(y_t, y_p))
    