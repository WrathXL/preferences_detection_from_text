from corpus_process import Corpus
from torch.utils.data import DataLoader, random_split
from transformers import BertForTokenClassification, AdamW
import torch
from seqeval.metrics import accuracy_score, f1_score, classification_report
from seqeval.metrics import classification_report



files_train_paths = ['data/sensitive1.tsv', 'data/sensitive3.tsv']
files_test_path = ['data/sensitive2.tsv']
pretrained_dataset = 'bert-base-uncased'

# ========================================
#               DATA
# ========================================

data_train = Corpus(files_train_paths).get_dataset_bert()
data_test = Corpus(files_test_path).get_dataset_bert()

val_size = int(0.4 * len(data_train))
train_size = len(data_train) - val_size


train_dataset, val_dataset = random_split(data_train, [train_size, val_size])

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print("train size : {}".format(train_size))
print("val size : {}".format(val_size))

# ========================================
#               MODEL
# ========================================

LABELS = Corpus(files_test_path).labels

model = BertForTokenClassification.from_pretrained(pretrained_dataset, num_labels=len(LABELS))
optimizer = AdamW(model.parameters(),  lr = 2e-5, eps = 1e-8)

# print functions
def p_epoch_status(epoch, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')

def p_batch_staus(step, size, epoch):
    print(
        'Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
            epoch, step * size, len(train_loader.dataset),
            100. * step / len(train_loader)))


def train(epochs=5):

    for epoch in range(epochs):
        print(p_epoch_status(epoch, epochs))
        # ========================================
        #               Training
        # ========================================
        total_train_loss = 0
        model.train()
        step = 1
        for input_ids, labels, mask in train_loader:

            p_batch_staus(step, len(input_ids), epoch)
            step += 1

            model.zero_grad()

            output = model(input_ids, token_type_ids=None,
                     attention_mask=mask, labels=labels)

            loss = output[0]
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print("Average train loss: {}".format(avg_train_loss))


        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        model.eval()
        total_val_loss = 0

        pred_labels = []
        true_labels = []

        for input_ids, labels, mask in val_loader:

            with torch.no_grad():
                output = model(input_ids, token_type_ids=None,
                     attention_mask=mask, labels=labels)

            loss, logits = output[:2]
            total_val_loss += loss.item()

            logits = logits.argmax(dim=2).view(-1)
            labels = labels.view(-1)
            mask = mask.view(-1)

            for i in range(len(input_ids)):
                if mask[i]:
                    pred_labels.append(LABELS[logits[i]])
                    true_labels.append(LABELS[labels[i]])

            print(mask)
            print(true_labels)
            print(logits)

        eval_loss = total_val_loss / len(val_loader)
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(accuracy_score(true_labels, pred_labels)))
        print("Validation F1-Score: {}".format(f1_score(true_labels, pred_labels)))
        print(classification_report(true_labels, pred_labels))

def test():
    pred_labels = []
    true_labels = []

    for input_ids, labels, mask in data_test:

        with torch.no_grad():
            logits = model(torch.tensor([input_ids.numpy()]))

        logits = logits[0].argmax(dim=2).view(-1)
        mask = mask.view(-1)

        for i in range(len(input_ids)):
            if mask[i]:
                pred_labels.append(LABELS[logits[i]])
                true_labels.append(LABELS[labels[i]])

    print("Test Accuracy: {}".format(accuracy_score(true_labels, pred_labels)))
    print("Test F1-Score: {}".format(f1_score(true_labels, pred_labels)))
    print(classification_report(true_labels, pred_labels))

    print(classification_report(true_labels, pred_labels))


if __name__ == '__main__':
    train(4)
    test()
    print(model)


# def eval(input_ids, mask):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#     toks = tokenizer.convert_ids_to_tokens(input_ids)
#     logits = model(torch.tensor([list(input_ids)]), token_type_ids=None ,attention_mask=torch.tensor([list(mask)]))
#     logits = logits.argmax(dim=2).view(-1)
#     labels = [corpus.labels[i] for i in logits]
#     return list(zip(toks, labels))



