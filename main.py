import pandas as pd

#parsing document to csv

class data:
    def __init__(self, path):
        self.csv = self._get_csv_data(path)
        self.size = self._get_size()
        self.labels = ['fact', 'preference', 'activity', 'subject']
        self.label_count = self._label_count()


    def _get_csv_data(self, path):
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


if __name__ == '__main__':
    d = data('data/sensitive-2.tsv')
