from corpus_process import Corpus
from torch.utils.data import DataLoader, random_split
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import torch
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from tqdm import tqdm, trange

files_paths = ['data/sensitive1.tsv', 'data/sensitive2.tsv', 'data/sensitive3.tsv']

# ========================================
#               DATA
# ========================================

corpus = Corpus(files_paths)

dataset = corpus.get_dataset_bert()


test_size = int(.3 * len(dataset))
rest = len(dataset) - test_size

val_size = int(.2 * rest)
train_size = rest - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size]) # !!! CAMBIAR SIZES

batch_size = 64 # !!!! CAMBIAR

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# ========================================
#               MODEL
# ========================================


model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(corpus.labels))
optimizer = BertAdam(model.parameters(),  lr = 2e-5, eps = 1e-8)


def flat_acc(pred, labels):
    pred = pred.argmax(dim=2).view(-1)
    labels = labels.view(-1)

    total, accept = 0, 0

    for i, l in enumerate(labels):
        if l == 0:
            continue
        total += 1
        if l == pred[i]:
            accept += 1

    return accept / total

def acc_by_label(pred, labels):
    pred = pred.argmax(dim=2).view(-1)
    labels = labels.view(-1)

    count = torch.zeros(11)
    accept = torch.zeros(11)

    for i, l in enumerate(labels):
        count[l] += 1
        if l == pred[i]:
            accept[l] += 1

    return count, accept

def train(epochs=5):

    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')

        # ========================================
        #               Training
        # ========================================
        total_train_loss = 0
        total_train_acc = 0
        model.train()

        step = 1
        for input_ids, labels, mask in train_loader:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]'.
                    format(epoch, step * len(input_ids), len(train_loader.dataset),
                           100. * step / len(train_loader)))


            #print("batch {}".format(step))
            step += 1
            model.zero_grad()

            loss = model(input_ids, token_type_ids=None,
                     attention_mask=mask, labels=labels)

            total_train_loss += loss.item()

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

        model.eval()
        total_val_loss = 0
        total_val_acc = 0

        count = torch.zeros(11)
        accept = torch.zeros(11)

        pred_labels = []
        true_labels = []

        for input_ids, labels, mask in val_loader:

            with torch.no_grad():
                logits = model(input_ids, token_type_ids=None,
                     attention_mask=mask)
                #loss = model(input_ids, token_type_ids=None,
                #    attention_mask=mask, labels=labels)

                #total_val_loss += loss.item()
                #total_val_acc += flat_acc(logits, labels)

                #x, y = acc_by_label(logits, labels)
                #count += x
                #accept += y

            #labels [batch, 79]
            #logits [batch, 79, 11]
            #Preparing data for metrics
            logits = logits.argmax(dim=2).view(-1)
            labels = labels.view(-1)
            input_ids = input_ids.view(-1)
            for i in range(len(input_ids)):
                if input_ids[i] != 0:
                    pred_labels.append(corpus.labels[logits[i]])
                    true_labels.append(corpus.labels[labels[i]])


        #print("  Accuracy: {0:.2f}".format(total_val_acc / len(val_loader)))
        #print("  Validation Loss: {0:.2f}".format(total_val_loss / len(val_loader)))
        #print(" Acc by labels {}".format(list(accept / count)))
        print(true_labels)
        print(pred_labels)
        print(classification_report(true_labels, pred_labels))
        print("Validation accuaracy: {}", accuracy_score(true_labels, pred_labels))

def test():
    pred_labels = []
    true_labels = []

    for input_ids, labels, _ in test_dataset:
        input_ids = input_ids.numpy()
        with torch.no_grad():
            logits = model(torch.tensor([input_ids]))

        logits = logits.argmax(dim=2).view(-1)

        for i in range(len(input_ids)):
            if input_ids[i] != 0:
                pred_labels.append(corpus.labels[logits[i]])
                true_labels.append(corpus.labels[labels[i]])

    print("\nTest results\n")
    print(classification_report(true_labels, pred_labels))
    print("Validation accuaracy:", accuracy_score(true_labels, pred_labels))


if __name__ == '__main__':
    train(4)
    test()


# def eval(input_ids, mask):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#     toks = tokenizer.convert_ids_to_tokens(input_ids)
#     logits = model(torch.tensor([list(input_ids)]), token_type_ids=None ,attention_mask=torch.tensor([list(mask)]))
#     logits = logits.argmax(dim=2).view(-1)
#     labels = [corpus.labels[i] for i in logits]
#     return list(zip(toks, labels))



