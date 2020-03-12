import numpy as np
import json
from configs import parameters, paths

class corpus:
    '''
    Brinda los métodos necesarios para procesar un '.tsv' exportado por el
    programa de anotación 'webanno' y retornar los tensores necesarios para distintos
    modelos
    '''

    def __init__(self):
        with open(paths.vocabulary_path) as fd:
            self.vocabulary = json.load(fd)
        self.tags = {l : parameters.labels.index(l) for l in parameters.labels}

    def read_tsv(self, paths, lower=True, bio=True):
        '''
        Lee  archivos '.tsv' exportado por webanno, limpia el texto y retorna dos numpy d-array
        en paralelo (sentences, tags)

        sentences[i] es la lista de los indeces de los tokens correspondientes a la oración
        i en el vocabulario de la clase

        tags[i] es la lista de etiquetas correspondientes a los tokens de la oración i
        '''

        text = []
        for path in paths:
            with open(path) as fd:
                text += fd.readlines()

        text = self._clean_text(text)
        #text[i] es de esta forma ['1-1', 'the', '_']

        x, y = self._sent_and_tags(text)

        return x, y

    def index_of_word(self, word):
        '''
        devuelve el indice de una palabra en el vocabulario definido para la clase
        '''
        return self.vocabulary[word] if word in self.vocabulary else -1

    def index_of_tag(self, tag):
        '''
        devuelve el indice de una etiqueta en las etiquetas definidas para la clase
        '''
        return  self.tags[tag] if tag in self.tags else -1


    def _clean_text(self, text):
        # remove comments
        text = filter(lambda s: not s.startswith('#'), text)
        # split by tabs
        text = [line.split('\t') for line in text]
        # remove endlines
        for lis in text: lis.pop()
        # remove empty lists
        text = list(filter(lambda l: l, text))
        #remove unnecessary columns
        text = [[line[0], line[2], line[4]] for line in text]
        return text

    def _sent_and_tags(self, text):
        # text[i] es de esta forma ['1-1', 'the', '_']
        max_len = max([int(toks[0].split('-')[1]) for toks in text])
        count = int(text[-1][0].split('-')[0])

        sent = [-1] * np.ones((count, max_len))
        tag = [-1] * np.ones((count, max_len))

        for line in text:
            id_sent, id_tok = map(int, line[0].split('-'))
            sent[id_sent - 1, id_tok - 1] = self.index_of_word(line[1])
            label = -1
            if line[2] != -1:
                bio = label[2][len(label[2]) - 2]
                label = bio + '-' + label[2][:len(label[2]) - 3]
            tag[id_sent - 1, id_tok - 1] = self.index_of_tag(label)

        return sent, tag


if __name__ == '__main__':
    files = ['data/sensitive-3.tsv']
    c = corpus()
    data = c.read_tsv(files)
