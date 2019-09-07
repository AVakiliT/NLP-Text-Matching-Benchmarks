import math
import random

import gensim
from collections import Counter
from itertools import chain
import numpy as np
import pandas as pd
import torch
from colorama import Fore
from nltk import word_tokenize
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import nn
from torch.nn import functional as F, init
from tqdm import tqdm

# %%
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# %%
MIN_FREQ = 2
tokenize = lambda x: x.split()
EMBEDDING_DIM = 200
PAD = 1
UNK = 0
device = torch.device('cuda')
EPSILON = 1e-8
INF = 1e13
PAD_FIRST = False

# N_EPOCH = 10
# BATCH_SIZE = 256
# MAX_LEN = 25
# FIX_LEN = False
# DATASET_NAME = 'quora'
# NUM_CLASS = 2
# KEEP = None
# LR = 10e-4


# N_EPOCH = 10
# BATCH_SIZE = 256
# MAX_LEN = 80
# FIX_LEN = False
# DATASET_NAME = 'SciTailV1.1'
# NUM_CLASS = 2
# KEEP = None
# LR = 2e-4


N_EPOCH = 10
BATCH_SIZE = 32
MAX_LEN = 45
FIX_LEN = False
DATASET_NAME = 'snli'
NUM_CLASS = 3
KEEP = None
LR = 10e-4

# %%
def read_train(fname):
    df = pd.read_csv(fname, index_col=None)
    text = df.apply(lambda row: ['<CLS>'] + tokenize(row[0]) + ['<SEP>'] + tokenize(row[1]) + ['<SEP>'], axis=1)
    texts_counter = Counter(chain.from_iterable(text))

    unk_count = 0
    for w, _ in texts_counter.most_common():
        f = texts_counter[w]
        if f < MIN_FREQ:
            unk_count += f
            texts_counter.pop(w)
    words_set = [i[0] for i in texts_counter.most_common()]
    itos = ['<UNK>', '<PAD>'] + words_set
    stoi = {v: i for i, v in enumerate(itos)}
    text_idx = list(map(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]), text))

    labels = df.iloc[:, 2].tolist()
    return stoi, itos, text_idx, labels


def read_eval(fname, stoi):
    df = pd.read_csv(fname)
    text = df.apply(lambda row: ['<CLS>'] + tokenize(row[0]) + ['<SEP>'] + tokenize(row[1]) + ['<SEP>'], axis=1)

    text_idx = list(map(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]), text))

    labels = df.iloc[:, 2]
    return text_idx, labels


class CustomDataset(Dataset):

    def __init__(self, context_idx, labels, keep=None) -> None:
        super().__init__()
        if keep is not None:
            index = labels.sample(keep).index
            context_idx = context_idx[index].reset_index(drop=True)

            labels = labels[index].reset_index(drop=True)

        self.contexts = context_idx

        self.labels = labels

    def __getitem__(self, index: int):
        return self.contexts[index], self.labels[index]

    def __len__(self) -> int:
        return self.labels.__len__()


print('Reading Dataset {} ...'.format(DATASET_NAME))
stoi, itos, context_idx, labels = read_train(
    DATASET_NAME + '/train.csv')
train_dataset = CustomDataset(context_idx, labels, keep=KEEP)
dev_dataset = CustomDataset(*read_eval(DATASET_NAME + '/dev.csv', stoi))
test_dataset = CustomDataset(*read_eval(DATASET_NAME + '/test.csv', stoi))


# %%

# TODO sort batches

def pad(s, l):
    if PAD_FIRST:
        return [PAD] * (l - len(s)) + s
    else:
        return s + [PAD] * (l - len(s))


def collate(batch):
    # m = max([len(i[0]) for i in batch])
    if FIX_LEN:
        m = MAX_LEN
    else:
        m = np.array([len(i[0]) for i in batch]).max().tolist()
    contexts = torch.LongTensor([pad(item[0], m) for item in batch])
    labels = torch.LongTensor([item[1] for item in batch])
    return [contexts, labels]


train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dev_data_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

VOCAB_LEN = len(itos)
print('Vocab length is {}'.format(VOCAB_LEN))

# %%
print('Reading Embeddings...')
w2v = gensim.models.KeyedVectors.load_word2vec_format(
    'embeddings/glove.6B.' + str(EMBEDDING_DIM)
    + 'd.w2vformat.bin',
    binary=True)

embedding_weights = torch.zeros(VOCAB_LEN, EMBEDDING_DIM)
nn.init.normal_(embedding_weights)

unmatch = []
for i, word in enumerate(itos):
    if word in w2v and i != PAD:
        embedding_weights[i] = torch.Tensor(w2v[word])
    else:
        unmatch.append(word)
        if i == PAD:
            embedding_weights[i] = torch.zeros(EMBEDDING_DIM)

print('{:.2f}% of words not in embedding file.'.format(len(unmatch) * 100 / VOCAB_LEN))


def get_emb():
    if True:
        emb = nn.Embedding(VOCAB_LEN, EMBEDDING_DIM, padding_idx=PAD, _weight=embedding_weights.clone())
        # emb.weight.requires_grad = False
        return emb
    else:
        return nn.Embedding(VOCAB_LEN, EMBEDDING_DIM, padding_idx=PAD)


# %%

class DiSAN(nn.Module):

    def __init__(self, dim=EMBEDDING_DIM):
        super().__init__()
        self.emb = get_emb()
        self.dim = dim
        self.Wh = nn.Linear(dim, dim)
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))
        nn.init.uniform_(self.b, -1 / math.sqrt(dim), 1 / math.sqrt(dim))
        self.c = nn.Parameter(torch.Tensor([5.0]))
        self.rnn = nn.LSTM(dim, dim // 2, bidirectional=True, batch_first=True)

        self.Wf1 = nn.Linear(dim, dim, bias=False)
        self.Wf2 = nn.Linear(dim, dim)
        self.Ws1 = nn.Linear(2 * dim, 2 * dim)
        self.Ws = nn.Linear(2 * dim, 2 * dim)

        self.final = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def get_masks(self, max_len):
        fw_mask = torch.BoolTensor(np.tri(max_len, max_len, dtype='bool')) \
            .unsqueeze(-1).expand(max_len, max_len, self.dim).to(device)
        bw_mask = torch.BoolTensor(np.tri(max_len, max_len, dtype='bool')).t() \
            .unsqueeze(-1).expand(max_len, max_len, self.dim).to(device)
        return fw_mask, bw_mask

    def multi_dim_masked_attention(self, h, att, m):
        att = att.masked_fill(m, -INF)
        att = F.softmax(att, -2)  # BLLE
        s = torch.einsum('bme,blme->ble', [h, att])  # BLE
        f = torch.sigmoid(self.Wf1(s) + self.Wf2(h))  # BLE
        u = f * h + (1 - f) * s
        return u

    def forward(self, x):
        xx = self.emb(x)  # BLE
        mask = xx == PAD
        h = F.elu(self.Wh(xx))
        # h = self.rnn(xx)[0]
        h1 = self.W1(h)
        h2 = self.W2(h)
        att = self.c * torch.tanh(((h1.unsqueeze(2) + h2.unsqueeze(1)) + self.b) / self.c)  # BLLE
        mask_2d = (mask.unsqueeze(1) | mask.unsqueeze(2))  # LL1
        att = att.masked_fill(mask_2d, -INF)  # BLLE
        max_len = xx.shape[1]
        fw_mask, bw_mask = self.get_masks(max_len)
        u_fw = self.multi_dim_masked_attention(h, att, fw_mask)  # BLE
        u_bw = self.multi_dim_masked_attention(h, att, bw_mask)  # BLE
        u = torch.cat([u_fw, u_bw], -1)  # BL(2E)

        att_s = self.Ws(F.elu(self.Ws1(u)))  # BL(2E)
        s_s = (u * att_s).sum(-2)  # B(2E)

        y_hat = self.final(s_s).squeeze(-1)
        return y_hat


class BiLSTM(nn.Module):

    def __init__(self, dim=EMBEDDING_DIM) -> None:
        super().__init__()
        self.emb = get_emb()
        self.RNN1 = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.RNN2 = nn.LSTM(dim * 2 * 4, dim * 2 * 4, batch_first=True, bidirectional=True)
        self.final = nn.Sequential(
            nn.Linear(dim * 2 * 4 * 2 * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1 if NUM_CLASS == 2 else NUM_CLASS)

        )

    def forward(self, x):
        mask = x == PAD
        xx = self.emb(x)
        xh = self.RNN1(xx)[0]

        mask2d = mask.unsqueeze(1) | mask.unsqueeze(2) | torch.diag(torch.ones(x.shape[1])).bool().unsqueeze(0).to(device)
        attn = torch.einsum('bmh,bnh->bmn', [xh, xh])
        attn = attn.masked_fill(mask2d, -INF)

        xhat = torch.einsum('bmn,bmh->bnh', [attn.softmax(1), xh])

        xc = torch.cat([xhat, xh, xhat - xh, xhat * xh], -1)
        xc = self.RNN2(xc)[0]

        v = torch.cat([xc.max(1)[0], xc[:, 0, :]], -1)
        # v = xc[:, 0, :]
        y_hat = self.final(v).squeeze(-1)

        return y_hat


# %%
# model = DiSAN()
model = BiLSTM()

model = model.to(device)


def calc_confusion_matrix(y, y_hat, num_classes=2):
    with torch.no_grad():
        if num_classes == 2:
            pred = y_hat.gt(0)
        else:
            pred = y_hat.argmax(-1)

    m = confusion_matrix(y.cpu(), pred.cpu(), range(num_classes))
    return m


def calc_metrics(m):
    p = (m.diagonal() / m.sum(0).clip(EPSILON)).mean()
    r = (m.diagonal() / m.sum(1).clip(EPSILON)).mean()
    f1 = ((2 * p * r) / (p + r)).mean()
    accu = m.diagonal().sum() / m.sum()
    return p, r, f1, accu


# optimizer = torch.optim.Adamax(params=model.parameters())
optimizer = torch.optim.Adam(lr=LR, params=model.parameters())

# train_weights = (1 / train_dataset.labels.value_counts(normalize=True))
# train_weights = train_weights.to_numpy() / train_weights.sum()
# train_weights = torch.Tensor(train_weights).to(device)

# progress_bar = tqdm(range(1, N_EPOCH + 1))
# for i_epoch in progress_bar:
for i_epoch in range(1, N_EPOCH + 1):
    model.train()
    loss_total = 0
    # accu_total = 0
    total = 0
    train_metrics = np.zeros((NUM_CLASS, NUM_CLASS))

    progress_bar = tqdm(train_data_loader)
    for x, y in progress_bar:
        # for x1, x2, y in train_data_loader:
        optimizer.zero_grad()
        # weights = train_weights[y]
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        if NUM_CLASS == 2:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction='sum')
        else:
            loss = F.cross_entropy(y_hat, y, reduction='sum')

        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        total += y.shape[0]
        train_metrics += calc_confusion_matrix(y, y_hat, NUM_CLASS)

        metrics = (
            loss_total / total,
            *calc_metrics(train_metrics),
        )

        progress_bar.set_description(Fore.RESET + '[ EP {0:02d} ]'
        # '[ TRN LS: {1:.4f} PR: {2:.4f} F1: {4:.4f} AC: {5:.4f}]'
                                                  '[ TRN LS: {1:.4f} AC: {5:.4f} ]'
                                     .format(i_epoch, *metrics))

    model.eval()

    loss_total_dev = 0
    total_dev = 0
    dev_metrics = np.zeros((NUM_CLASS, NUM_CLASS))

    with torch.no_grad():
        progress_bar = tqdm(dev_data_loader)
        for x, y in progress_bar:
            # for x1, x2, y in dev_data_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            if NUM_CLASS == 2:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction='sum')
            else:
                loss = F.cross_entropy(y_hat, y, reduction='sum')

            loss_total_dev += loss.item()
            total_dev += y.shape[0]
            dev_metrics += calc_confusion_matrix(y, y_hat, NUM_CLASS)

            metrics = (
                loss_total_dev / total_dev,
                *calc_metrics(dev_metrics)
            )

            progress_bar.set_description(Fore.BLUE + '[ EP {0:02d} ]'
            # '[ TST LS: {1:.4f} PR: {2:.4f} F1: {4:.4f} AC: {5:.4f} ]'
                                                     '[ DEV LS: {1:.4f} AC: {5:.4f} ]'
                                         .format(i_epoch, *metrics))

    # model.eval()
    #
    # loss_total_test = 0
    # total_test = 0
    # test_metrics = np.zeros((NUM_CLASS, NUM_CLASS))
    #
    # with torch.no_grad():
    #     progress_bar = tqdm(test_data_loader)
    #     for x1, x2, y in progress_bar:
    #         # for x1, x2, y in dev_data_loader:
    #         x1, x2, y = x1.to(device), x2.to(device), y.to(device)
    #         y_hat = model(x1, x2)
    #
    #         if NUM_CLASS == 2:
    #             loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction='sum')
    #         else:
    #             loss = F.cross_entropy(y_hat, y, reduction='sum')
    #
    #         loss_total_test += loss.item()
    #         total_test += y.shape[0]
    #         test_metrics += calc_confusion_matrix(y, y_hat, NUM_CLASS)
    #
    #         metrics = (
    #             loss_total_test / total_test,
    #             *calc_metrics(test_metrics)
    #         )
    #
    #         progress_bar.set_description(Fore.YELLOW + '[ EP {0:02d} ]'
    #         # '[ TST LS: {1:.4f} PR: {2:.4f} F1: {4:.4f} AC: {5:.4f} ]'
    #                                                    '[ TST LS: {1:.4f} AC: {5:.4f} ]'
    #                                      .format(i_epoch, *metrics))

    # metrics = (
    #     loss_total / total,
    #     *p_r_(train_metrics),
    #     loss_total_test / total_test,
    #     *p_r_(test_metrics)
    # )
    #
    # progress_bar.set_description('[ EP {0:02d} ]'
    #                              # '[ TRN LS: {1:.4f} PR: {2:.4f} F1: {3:.4f} AC: {4:.4f}]'
    #                              '[ TRN LS: {1:.4f} AC: {5:.4f} ]'
    #                              # '[ TST LS: {5:.4f} PR: {6:.4f} F1: {7:.4f} AC: {8:.4f} ]'
    #                              '[ TST LS: {6:.4f} AC: {10:.4f} ]'
    #                              .format(i_epoch, *metrics))
