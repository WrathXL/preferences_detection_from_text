import pandas as pd
import numpy as np
import json

class data:
    def __init__(self, path_data, path_vocabulary):
        self.csv = self._get_csv_data(path_data) # Actually a DataFrame
        self.size = self._get_size()
        self.labels = ['fact', 'preference', 'activity', 'subject', 'event']
        self.label_count = self._label_count()
        self.tags = []

        with open(path_vocabulary) as fd:
            self.vocabulary = json.load(fd)


    def _get_csv_data(self, path):
        '''
        :param path: The .tsf webanno file exported by annotations
        :return: A pandas DataFrames containing the tokens and labels after the following processing
        * Remove comments
        * Remove endlines
        * Remove useless data
        '''
        with open(path) as fd:
            text = fd.readlines()

        text = filter(lambda s: not s.startswith('#'), text) #remove comments
        text = [line.split('\t') for line in text] #split by tabs
        for lis in text: lis.pop() #remove endlines
        text = list(filter(lambda l : l, text)) #remove empty lists
        return pd.DataFrame(text, columns=['sent-token', 'letters', 'word', 'BIO', 'label'])

    def _get_size(self):
        return max([int(s.split('-')[0]) for s in self.csv['sent-token']])

    def _label_count(self):
        labels = {l : 0 for l in self.labels}

        s = set()
        for i in self.csv['label']:
            if i and i.split('[')[0] in self.labels:
                s.add(i)

        for l in s:
            labels[l.split('[')[0]] += 1

        return labels

    def lower_words(self):
        '''
        Convert all words to lower case
        '''
        for i, word in enumerate(self.csv['word']):
            self.csv['word'][i] = self.csv['word'][i].lower()

    def build_bio(self):
        '''
        Automatic filling of BIO tagging in a dataframe column
        '''
        s = set()
        for i, l in enumerate(self.csv['label']):
            if l and l.split('[')[0] in self.labels:
                if l not in s:
                    s.add(l)
                    self.csv['BIO'][i] = 'B'
                else:
                    self.csv['BIO'][i] = 'I'
            else:
                self.csv['BIO'][i] = 'O'

    def automatic_subject_tagging(self):
        '''
        Automatic tagging of words that can be subjects in the meaning of the schema
        '''
        keywords = ['i', 'she', 'he', 'we', 'they']
        for i, w in enumerate(self.csv['word']):
            if w in keywords :
                self.csv['label'][i] = 'subject[' + str(self.size + i) + ']'
                self.label_count['subject'] += 1
                self.csv['BIO'][i] = 'B'


    def sentence_and_labels(self):
        self.lower_words()
        self.automatic_subject_tagging()
        self.build_bio()

        sentences = []
        tags = []

        all_tags = set()

        cur_sent = []
        cur_tag = []
        piv = 1

        for i in range(len(self.csv)):
            sent = int(self.csv['sent-token'][i].split('-')[0])
            word = self.csv['word'][i]
            tag = self.csv['BIO'][i]
            if tag != 'O':
                tag += '-' + self.csv['label'][i].split('[')[0]

            if sent != piv:
                piv += 1
                sentences.append(cur_sent)
                tags.append(cur_tag)
                cur_sent, cur_tag = [], []


            if word in self.vocabulary:
                cur_sent.append(self.vocabulary[word])
            else:
                cur_sent.append(-1)

            cur_tag.append(tag)
            all_tags.add(tag)

        all_tags = list(all_tags)
        tags = [list(map(lambda x : all_tags.index(x), sen)) for sen in tags]

        max_len = max([len(sent) for sent in tags])

        x = [-1] * np.ones((len(tags), max_len))
        y = [-1] * np.ones((len(tags), max_len))

        for i in range(len(sentences)):
            l = len(sentences[i])
            x[i][:l] = sentences[i]
            y[i][:l] = tags[i]

        return x, y



if __name__ == '__main__':
    d = data('data/sensitive-2.tsv', 'data/vocabulary.json')
    x, y = d.sentence_and_labels()