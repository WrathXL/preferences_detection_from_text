import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding import pretrained_embedding



class ner_simple(nn.Module):

    def __init__(self, embedding_path=None):
        super(ner_simple, self).__init__()

        #e = pretrained_embedding(embedding_path)

        #self.embedding = nn.Embedding(e.vocabulary_size, e.dimension)

        #self.embedding.weight.data.copy_(torch.from_numpy(e.word_vecs))
        #self.embedding.weight.requires_grad = False

        self.embedding = nn.Embedding(4031, 64)
        self.lstm = nn.LSTM(64, 64, bidirectional=True)
        self.fc = nn.Linear(128, 11)

    def forward(self, x):

        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.view(-1, x.shape[2])
        x = self.fc(x)

        return F.softmax(x, dim=1)

    def loss_fn(outputs, labels):
        labels = labels.view(-1)
        mask = (labels >= 0).float()
        num_tokens = int(torch.sum(mask).item())

        outputs = outputs[range(outputs.shape[0]), labels] * mask
        return -torch.sum(outputs) / num_tokens

    def accuaracy(outputs, labels):
        labels = labels.view(-1)
        mask = (labels >= 0).float()
        num_tokens = int(torch.sum(mask).item())

        predictions = torch.argmax(outputs, 1)

        return torch.sum(predictions == labels).item() / num_tokens


if __name__ == '__main__':
    embedding_path = 'data/glove.6B.100d.txt'
