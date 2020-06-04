import torch
import json
from torch.utils.data import TensorDataset
from transformers import BertTokenizer


class Corpus:
    '''
    Brinda los métodos necesarios para procesar un '.tsv' exportado por el
    programa de anotación 'webanno' y retornar los tensores necesarios para distintos
    modelos
    '''

    def __init__(self, files, vocabulary_path=None):

        self.files_paths = files

        if vocabulary_path:
            with open(vocabulary_path) as fd:
                self.vocabulary = json.load(fd)

        self.labels = ['O', 'B-subject', 'I-subject', 'B-preference', 'I-preference', 'B-fact', 'I-fact', 'B-activity', 'I-activity', 'B-object', 'I-object']
        self.labels_dict = {l : self.labels.index(l) for l in self.labels}

        self.subjects = ['i', 'we', 'he', 'she', 'they']

    def read_tsv(self,  lower=True, auto_subject=True):
        '''
        Lee  archivos '.tsv' exportado por webanno, limpia el texto y retorna dos numpy d-array
        en paralelo (sentences, tags)

        sentences[i] es la lista de los indeces de los tokens correspondientes a la oración
        i en el vocabulario de la clase

        tags[i] es la lista de etiquetas correspondientes a los tokens de la oración i
        '''

        text = []
        for path in self.files_paths:
            with open(path) as fd:
                text += fd.readlines()

        text = self._clean_text(text)
        #text[i] es de esta forma ['1-1', 'the', '_']

        x, y = self._sent_and_tags(text)

        if lower:
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j] = x[i][j].lower()

        if auto_subject:
            for i in range(len(x)):
                for j in range(len(x[i])):
                    if x[i][j] in self.subjects and y[i][j] == 'O':
                        y[i][j] = 'B-subject'

        return x, y

    def index_of_word(self, word):
        '''
        devuelve el indice de una palabra en el vocabulario definido para la clase
        '''
        return self.vocabulary[word] + 1 if word in self.vocabulary else 0

    def index_of_label(self, tag):
        '''
        devuelve el indice de una etiqueta en las etiquetas definidas para la clase
        '''
        return  self.labels_dict[tag] if tag in self.labels_dict else 0


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
        text = [[line[0], line[2], line[4], line[3]] for line in text]
        return text

    def _sent_and_tags(self, text):
        # text[i] es de esta forma ['1-1', 'the', '_']

        sent, tag = [], []

        cur_sent, cur_tag = [], []
        piv_id = 1

        for i, line in enumerate(text):
            id_sent, id_tok = map(int, line[0].split('-'))

            if id_sent != piv_id:
                if len(cur_sent) > 1:
                    sent.append(cur_sent)
                    tag.append(cur_tag)
                cur_sent, cur_tag = [], []
                piv_id = id_sent

            cur_sent.append(line[1])

            if line[2] != '_':
                if line[3] == text[i - 1][3] and line[3] != '*':
                    cur_tag.append('I-' + line[2].split('[')[0] )
                else:
                    cur_tag.append('B-' + line[2].split('[')[0])
            else:
                cur_tag.append('O')


        return sent, tag



    def get_dataset(self):
        '''
        :return: Dataset torch object for models
        '''
        sent, tags = self.read_tsv()

        max_len = max([len(s) for s in sent])

        x = torch.full((len(sent), max_len), 0, dtype=torch.long)
        #loss functions needs to ignore y[i] == -1. Otherwise this x[i] == 0 is an index for a word in embedding
        y = torch.full((len(tags), max_len), -1, dtype=torch.long)

        for i, s in enumerate(sent):
            for j, tok in enumerate(s):
                x[i, j] = self.index_of_word(tok)

        for i, s in enumerate(tags):
            for j, t in enumerate(s):
                y[i, j] = self.index_of_label(t);


        return  TensorDataset(x, y)


    def get_dataset_bert(self):
        sents, tags = self.read_tsv()

        # Primero arrglar problema de la tokenización
        new_sents, new_tags = [], []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        for i, sent in enumerate(sents):
            new_sent = []
            new_tag = []
            for idx, tok in enumerate(sent):
                tag_pref = tags[i][idx].split('-')[0]

                for j, t in enumerate(tokenizer.tokenize(tok)):
                    new_sent.append(t)

                    if j > 0 and tag_pref == 'B':
                        new_tag.append('I-' + tags[i][idx].split('-')[1])
                    else:
                        new_tag.append(tags[i][idx])

            new_sents.append(new_sent)
            new_tags.append(new_tag)

        # Segundo padear las listas y crear los tensores

        max_len = max([len(s) for s in new_sents])

        x = torch.zeros((len(new_sents), max_len + 1), dtype=torch.long)
        y = torch.zeros((len(new_tags), max_len + 1), dtype=torch.long)
        masks = torch.zeros((len(new_sents), max_len + 1), dtype=torch.float)

        for i, sent in enumerate(new_sents):
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_len + 1,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            x[i] = encoded_dict['input_ids']
            y[i][1: len(sent) + 1] =  torch.tensor([self.index_of_label(l) for l in new_tags[i]])
            masks[i] = encoded_dict['attention_mask']

        return TensorDataset(x, y, masks)


    def build_sent(self, triplet):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        sent = tokenizer.convert_ids_to_tokens(triplet[0])
        tags = [self.labels[t] for t in triplet[1]]

        return list(zip(sent, tags))

if __name__ == '__main__':
    files = ['data/sensitive.tsv']
    voc_path = 'data/vocabulary_glove.json'



    c = Corpus(files, vocabulary_path=voc_path)
    dataset = c.get_dataset_bert()

    # sents, tags = c.read_tsv(files)
    # x = c.get_dataset(sents, tags);