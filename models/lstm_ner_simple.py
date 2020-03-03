import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import pretrained_embedding

class ner_simple(nn.Module):

    def __init__(self, params, embedding_path):
        super(ner_simple, self).__init__()

        self.embedding = pretrained_embedding(embedding_path)
        self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidd_dim, bidirectional=True)
        self.fc = nn.Linear(params.lstm_hidd_dim, params.numb_of_tags)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.view(-1, x.shape[2])
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

    def loss_function(self, outputs, labels):
        labels = labels.view(-1)
        mask = (labels >= 0).float()
        num_tokens = int(torch.sum(mask).data[0])
        outputs = outputs[range(outputs.shape[0]), labels] * mask
        return -torch.sum(outputs) / num_tokens





if __name__ == '__main__':
    embedding_path = 'data/glove.6B.100d.txt'
