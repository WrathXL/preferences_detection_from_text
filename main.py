from corpus_process import Corpus
from models.lstm_ner_simple import ner_simple
import torch
from torch.utils.data import DataLoader, random_split


files_paths = ['data/sensitive.tsv']
vocabulary_path = 'data/vocabulary_glove.json'
embedding_path = 'data/glove.6B.100d.txt'

# ========================================
#               DATA
# ========================================

corpus = Corpus(files_paths, vocabulary_path=vocabulary_path)

dataset = corpus.get_dataset()

val_size = int(.2 * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset,[train_size, val_size])

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ========================================
#               MODEL
# ========================================


model = ner_simple(embedding_path)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-8)




def train(epochs = 20):
    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')

        total_train_loss = 0
        total_train_acc = 0

        # ========================================
        #               Training
        # ========================================

        for input_ids, labels in train_loader:
            model.zero_grad()
            output = model(input_ids)
            loss = ner_simple.loss_fn(output, labels)

            total_train_loss += loss.item()
            total_train_acc += ner_simple.accuaracy(output, labels)

            loss.backward()


            optimizer.step()

        print("")
        print("  Average training loss: {0:.2f}".format(total_train_loss / len(train_loader)))
        print("  Average training acc : {0:.2f}".format(total_train_acc / len(train_loader)))


        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        total_val_loss = 0
        total_val_acc = 0

        for input_ids, labels in val_loader:

            with torch.no_grad():
                output = model(input_ids)
                loss = ner_simple.loss_fn(output, labels)

                total_val_loss += loss.item()
                total_val_acc += ner_simple.accuaracy(output, labels)

        print("  Accuracy: {0:.2f}".format(total_val_acc / len(val_loader)))
        print("  Validation Loss: {0:.2f}".format(total_val_loss / len(val_loader)))


train()

    

