import torch
from joblib import Parallel, delayed
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertModel, \
    DistilBertForSequenceClassification
from torch.utils.data.dataset import Dataset
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# %%
MIN_FREQ = 2
EMBEDDING_DIM = 200
device = torch.device('cuda')
EPSILON = 1e-8
INF = 1e13
PAD_FIRST = False

# N_EPOCH = 10
# BATCH_SIZE = 32
# MAX_LEN = 25
# FIX_LEN = True
# DATASET_NAME = 'quora'
# NUM_CLASS = 2

# N_EPOCH = 10
# BATCH_SIZE = 32
# MAX_LEN = 60
# FIX_LEN = False
# DATASET_NAME = 'SciTailV1.1'
# NUM_CLASS = 2
# KEEP = None
# LR = 2e-4


N_EPOCH = 10
BATCH_SIZE = 32
MAX_LEN = 30
FIX_LEN = False
DATASET_NAME = 'snli'
NUM_CLASS = 3
KEEP = None
LR = 10e-4


MAX_TOTAL_LEN = MAX_LEN * 2 + 1
# %%

def truncate(a):
    idx = a.index(SEP)
    a = a[0:min(MAX_LEN, idx)] + a[idx:idx + min(len(a) - idx, MAX_LEN + 1)]
    return a

class Ass(Dataset):
    def __init__(self, token_ids, labels):
        self.texts = token_ids
        self.labels = labels

    def __getitem__(self, index: int):
        return self.texts[index], self.labels[index]

    def __len__(self) -> int:
        return self.labels.__len__()



def read_data(fname, tokenizer):
    df = pd.read_csv(fname)

    df['text'] = df.apply(lambda row: '[CLS] ' + row[0] + ' [SEP] ' + row[1] + ' [SEP]', axis=1)

    tokenized_texts = df.text.apply(lambda x: tokenizer.tokenize(x))
    tokenized_texts = Parallel(n_jobs=16)(delayed(tokenizer.tokenize)(x) for x in tqdm(df.text[:100]))

    token_ids = tokenized_texts.apply(tokenizer.convert_tokens_to_ids)

    token_ids = token_ids.apply(truncate)

    token_ids_matrix = np.array(
        token_ids.apply(lambda x: x + [0] * max(0, MAX_TOTAL_LEN - len(x))).tolist()).astype('int64')

    sep_idx = token_ids_matrix.__eq__(SEP).argmax(1)
    mask_matrix = [[i < sep_idx[j] for i in range(MAX_TOTAL_LEN)] for j in range(len(token_ids_matrix))]
    mask_matrix = np.array(mask_matrix).astype('float')

    _dataset = TensorDataset(torch.tensor(token_ids_matrix), torch.tensor(mask_matrix),
                             torch.tensor(np.array(df.iloc[:, 2])))

    _data_loader = DataLoader(_dataset, batch_size=BATCH_SIZE)

    return _data_loader


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
PAD, SEP, CLS = tokenizer.vocab['[PAD]'], tokenizer.vocab['[SEP]'], tokenizer.vocab['[CLS]']

train_data_loader = read_data(DATASET_NAME + '/train.csv', tokenizer=tokenizer)
dev_data_loader = read_data(DATASET_NAME + '/dev.csv', tokenizer=tokenizer)
test_data_loader = read_data(DATASET_NAME + '/test.csv', tokenizer=tokenizer)


#%%

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=NUM_CLASS)
model = model.cuda()

#%%
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


optimizer = AdamW(model.parameters(), lr=2e-5)

#%%
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# trange is a tqdm wrapper around the normal python range
for ep in range(epochs):
    #   print('EPOCH', ep)

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    acc = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    p_bar =tqdm(train_data_loader)
    for step, batch in enumerate(p_bar):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss, logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        acc += logits.argmax(1).eq(b_labels).long().sum().item()
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1


        p_bar.set_description('Loss {:.4f} Acc {:.4f}'.format(tr_loss / nb_tr_steps, acc / nb_tr_examples))

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    #   print('EPOCH', ep)

    # Training

    with torch.no_grad():
        # Set our model to testing mode (as opposed to evaluation mode)
        model.eval()

        # Tracking variables
        ts_loss = 0
        acc = 0
        nb_ts_examples, nb_ts_steps = 0, 0


        # test the data for one epoch
        p_bar = tqdm(test_data_loader)
        for step, batch in enumerate(p_bar):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            # Forward pass
            loss, logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            acc += logits.argmax(1).eq(b_labels).long().sum().item()


            # Update tracking variables
            ts_loss += loss.item()
            nb_ts_examples += b_input_ids.size(0)
            nb_ts_steps += 1

            p_bar.set_description('Loss {:.4f} Acc {:.4f}'.format(ts_loss / nb_ts_steps, acc / nb_ts_examples))

        print("test loss: {}".format(ts_loss / nb_ts_steps))