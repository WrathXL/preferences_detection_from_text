from corpus_process import Corpus
from torch.utils.data import DataLoader, random_split
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
import torch
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter


files_train_paths = ['data/sensitive1.tsv', 'data/sensitive3.tsv']
files_test_path = ['data/sensitive2.tsv']
pretrained_dataset = 'bert-base-uncased'

# ========================================
#               DATA
# ========================================

data_train = Corpus(files_train_paths).get_dataset_bert()
data_test = Corpus(files_test_path).get_dataset_bert()

val_size = int(0.2 * len(data_train))
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


model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(LABELS))
optimizer = BertAdam(model.parameters(),  lr = 2e-5, eps = 1e-8)




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
                    pred_labels.append(LABELS[logits[i]])
                    true_labels.append(LABELS[labels[i]])


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

    for input_ids, labels, _ in data_test:
        input_ids = input_ids.numpy()
        with torch.no_grad():
            logits = model(torch.tensor([input_ids]))

        logits = logits.argmax(dim=2).view(-1)

        for i in range(len(input_ids)):
            if input_ids[i] != 0:
                pred_labels.append(LABELS[logits[i]])
                true_labels.append(LABELS[labels[i]])

    print("\nTest results\n")
    print(classification_report(true_labels, pred_labels))
    print("Validation accuaracy:", accuracy_score(true_labels, pred_labels))


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



