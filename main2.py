import math
import os
import pickle
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
from texttable import Texttable
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import nn
from torch.nn import functional as F, init
from tqdm import tqdm
from math import sqrt

# %%
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=100, type=int)
parser.add_argument('--embedding_version', default='6B')
parser.add_argument('--dropout', default=0., type=float)

# SciTail
# parser.add_argument('--epochs', default=10, type=int)
# parser.add_argument('--batch', default=128, type=int)
# parser.add_argument('--dataset', default='SciTailV1.1')
# parser.add_argument('--max_len', default=80)
# parser.add_argument('--num_class', default=2)
# parser.add_argument('--lr', default=1e-3)
# parser.add_argument('--min_freq', default=2)

# SNLI
parser.add_argument('--epochs', default=7, type=int)
parser.add_argument('--batch', default=128, type=int)
parser.add_argument('--dataset', default='snli')
parser.add_argument('--max_len', default=30)
parser.add_argument('--num_class', default=3)
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--min_freq', default=2)


# Quora
# parser.add_argument('--epochs', default=12, type=int)
# parser.add_argument('--batch', default=128, type=int)
# parser.add_argument('--dataset', default='quora')
# parser.add_argument('--max_len', default=25)
# parser.add_argument('--num_class', default=2)
# parser.add_argument('--lr', default=1e-3)
# parser.add_argument('--min_freq', default=15)

args = parser.parse_known_args()
print(args)

# %%


SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# %%
MIN_FREQ = args[0].min_freq
tokenize = lambda x: x.split()
EMBEDDING_DIM = args[0].embedding_dim
DROPOUT = args[0].dropout
GRAD_CLIP = 5
PAD = 1
UNK = 0
device = torch.device('cuda')
EPSILON = 1e-8
INF = 1e13
PAD_FIRST = False
KEEP = None
FIX_LEN = False

N_EPOCH = args[0].epochs
BATCH_SIZE = args[0].batch
MAX_LEN = args[0].max_len
DATASET_NAME = args[0].dataset
NUM_CLASS = args[0].num_class
LR = args[0].lr


def read_train(fname):
    df = pd.read_csv(fname, index_col=None)
    df.iloc[:, 0] = df.iloc[:, 0].apply(tokenize)
    df.iloc[:, 1] = df.iloc[:, 1].apply(tokenize)
    context_counter = Counter(chain.from_iterable(df.iloc[:, 0]))
    response_counter = Counter(chain.from_iterable(df.iloc[:, 1]))
    texts_counter = context_counter + response_counter
    unk_count = 0
    for w in list(texts_counter.keys()):
        f = texts_counter[w]
        if f < MIN_FREQ:
            unk_count += f
            texts_counter.pop(w)
    words_set = set(texts_counter.keys())
    itos = ['<UNK>', '<PAD>'] + list(words_set)
    stoi = {v: i for i, v in enumerate(itos)}
    context_idx = df.iloc[:, 0].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    response_idx = df.iloc[:, 1].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))

    char_counter = Counter(chain.from_iterable(map(list, chain.from_iterable(df.iloc[:, 0])))) + Counter(
        chain.from_iterable(map(list, chain.from_iterable(df.iloc[:, 1]))))
    char_itos = ['<UNK>', '<PAD>'] + [k for k, v in char_counter.most_common()]
    stoi_char = {v: i for i, v in enumerate(char_itos)}
    context_char_idx = df.iloc[:, 0].apply(lambda y: [[stoi_char[i] for i in x] for x in y][:MAX_LEN])
    response_char_idx = df.iloc[:, 1].apply(lambda y: [[stoi_char[i] for i in x] for x in y][:MAX_LEN])

    labels = df.iloc[:, 2]
    return stoi, itos, stoi_char, char_itos, context_idx, response_idx, context_char_idx, response_char_idx, labels


def read_eval(fname, stoi, stoi_char):
    df = pd.read_csv(fname)
    df.iloc[:, 0] = df.iloc[:, 0].apply(tokenize)
    df.iloc[:, 1] = df.iloc[:, 1].apply(tokenize)
    context_idx = df.iloc[:, 0].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    response_idx = df.iloc[:, 1].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))

    context_char_idx = df.iloc[:, 0].apply(
        lambda y: [[stoi_char[i] if i in stoi_char else UNK for i in x] for x in y][:MAX_LEN])
    response_char_idx = df.iloc[:, 1].apply(
        lambda y: [[stoi_char[i] if i in stoi_char else UNK for i in x] for x in y][:MAX_LEN])

    labels = df.iloc[:, 2]
    return context_idx, response_idx, context_char_idx, response_char_idx, labels


class CustomDataset(Dataset):

    def __init__(self, context_idx, response_idx, context_char_idx, response_char_idx, labels, keep=None) -> None:
        super().__init__()
        if keep is not None:
            index = labels.sample(keep).index
            context_idx = context_idx[index].reset_index(drop=True)
            context_char_idx = context_char_idx[index].reset_index(drop=True)
            response_idx = response_idx[index].reset_index(drop=True)
            response_char_idx = response_char_idx[index].reset_index(drop=True)
            labels = labels[index].reset_index(drop=True)

        self.contexts = context_idx
        self.contexts_char = context_char_idx
        self.responses = response_idx
        self.responses_char = response_char_idx
        self.labels = labels

    def __getitem__(self, index: int):
        return self.contexts[index], self.responses[index], self.labels[index]

    def __len__(self) -> int:
        return self.labels.__len__()


print('Reading Dataset {} ...'.format(DATASET_NAME))
stoi, itos, stoi_char, char_itos, context_idx, response_idx, context_char_idx, response_char_idx, labels = read_train(
    DATASET_NAME + '/train.csv')
train_dataset = CustomDataset(context_idx, response_idx, context_char_idx, response_char_idx, labels, keep=KEEP)
dev_dataset = CustomDataset(*read_eval(DATASET_NAME + '/dev.csv', stoi, stoi_char))
test_dataset = CustomDataset(*read_eval(DATASET_NAME + '/test.csv', stoi, stoi_char))


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
        m, n = MAX_LEN, MAX_LEN
    else:
        m, n = np.array([[len(i[0]), len(i[1])] for i in batch]).max(0).tolist()
    contexts = torch.LongTensor([pad(item[0], m) for item in batch])
    response = torch.LongTensor([pad(item[1], n) for item in batch])
    labels = torch.LongTensor([item[2] for item in batch])
    return [contexts, response, labels]


train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dev_data_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

VOCAB_LEN = len(itos)
print('Vocab length is {}'.format(VOCAB_LEN))

# %%
print('Reading Embeddings...')
w2v = gensim.models.KeyedVectors.load_word2vec_format(
    'embeddings/glove.' + args[0].embedding_version + '.' + str(EMBEDDING_DIM)
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
    class DiSAN_Block(nn.Module):

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

            self.Wf1 = nn.Linear(dim, dim, bias=False)
            self.Wf2 = nn.Linear(dim, dim)
            self.Ws1 = nn.Linear(2 * dim, 2 * dim)
            self.Ws = nn.Linear(2 * dim, 2 * dim)

        def multi_dim_masked_attention(self, h, att, m):
            att = att.masked_fill(m, -INF)
            att = F.softmax(att, -2)  # BLLE
            s = torch.einsum('bme,blme->ble', [h, att])  # BLE
            f = torch.sigmoid(self.Wf1(s) + self.Wf2(h))  # BLE
            u = f * h + (1 - f) * s
            return u

        def forward(self, x, mask):
            x = self.emb(x)  # BLE
            h = F.elu(self.Wh(x))
            h1 = self.W1(h)
            h2 = self.W2(h)
            att = self.c * torch.tanh(((h1.unsqueeze(2) + h2.unsqueeze(1)) + self.b) / self.c)  # BLLE
            mask_2d = (mask.unsqueeze(1).__or__(mask.unsqueeze(2))).unsqueeze(-1)  # LL1
            att = att.masked_fill(mask_2d, -INF)  # BLLE
            max_len = x.shape[1]
            fw_mask = torch.ByteTensor(np.tri(max_len, max_len, dtype='uint8')) \
                .unsqueeze(-1).expand(max_len, max_len, self.dim).to(device)
            bw_mask = torch.ByteTensor(np.tri(max_len, max_len, dtype='uint8')).t() \
                .unsqueeze(-1).expand(max_len, max_len, self.dim).to(device)
            u_fw = self.multi_dim_masked_attention(h, att, fw_mask)  # BLE
            u_bw = self.multi_dim_masked_attention(h, att, bw_mask)  # BLE
            u = torch.cat([u_fw, u_bw], -1)  # BL(2E)

            att_s = self.Ws(F.elu(self.Ws1(u)))  # BL(2E)
            s_s = (u * att_s).sum(-2)  # B(2E)
            return s_s

    def __init__(self):
        super().__init__()
        self.final = nn.Sequential(
            nn.Linear(2 * 4 * EMBEDDING_DIM, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, 1),
        )
        self.emb = get_emb()
        self.disan_block = DiSAN.DiSAN_Block()

    def forward(self, x1, x2):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD

        c = self.disan_block(x1, c_mask)
        r = self.disan_block(x2, r_mask)

        y_hat = self.final(torch.cat([c, r, c - r, c * r], -1)).squeeze(-1)

        return y_hat


class ESIM_(nn.Module):

    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, num_class=NUM_CLASS):
        super().__init__()
        self.emb = get_emb()
        self.RNN_1 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.RNN_2 = nn.LSTM(hidden_dim * 2 * 5, hidden_dim, batch_first=True, bidirectional=True)
        self.final = nn.Sequential(
            nn.Linear(2 * 2 * 5 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 if num_class == 2 else num_class),
        )

    def forward(self, x1, x2):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD

        xx1, xx2 = self.emb(x1), self.emb(x2)

        cs, _ = self.RNN_1(xx1)  # BL(2H)
        rs, _ = self.RNN_1(xx2)

        att = torch.einsum('bmh,bnh->bmn', [cs, rs])

        att_mask = c_mask.unsqueeze(2) | r_mask.unsqueeze(1)

        att = att.masked_fill(att_mask, -INF)

        c_att_scores = att.softmax(2)  # BMN
        c_hat = torch.einsum('bmn,bnh->bmh', [c_att_scores, rs])

        r_att_scores = att.softmax(1)  # BMN
        r_hat = torch.einsum('bmn,bmh->bnh', [r_att_scores, cs])

        c_m = torch.cat([cs, c_hat, cs - c_hat, cs * c_hat, (cs - c_hat) * (c_hat - cs)], -1)  # BL(8H)
        r_m = torch.cat([rs, r_hat, rs - r_hat, rs * r_hat, (rs - r_hat) * (r_hat - rs)], -1)

        c_m = c_m.masked_fill(c_mask.unsqueeze(-1), 0)
        r_m = r_m.masked_fill(r_mask.unsqueeze(-1), 0)

        c_v, _ = self.RNN_2(c_m)  # BL(16H)
        r_v, _ = self.RNN_2(r_m)

        c_v = c_v.masked_fill(c_mask.unsqueeze(-1), -INF)
        r_v = r_v.masked_fill(r_mask.unsqueeze(-1), -INF)

        c_v_max = c_v.max(1)[0]
        r_v_max = r_v.max(1)[0]

        c_v = c_v.masked_fill(c_mask.unsqueeze(-1), 0)
        r_v = r_v.masked_fill(r_mask.unsqueeze(-1), 0)

        c_v_mean = c_v.sum(1) / (~c_mask).sum(1, keepdim=True).float()
        r_v_mean = r_v.sum(1) / (~r_mask).sum(1, keepdim=True).float()

        vc = torch.cat([c_v_max, c_v_mean], -1)
        vr = torch.cat([r_v_max, r_v_mean], -1)
        v = torch.cat([vc, vr, vc - vr, vc * vr, (vc - vr) * (vr - vc)], -1)

        p = self.final(v).squeeze(-1)

        return p


class ESIM_SAN(nn.Module):

    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, num_class=NUM_CLASS):
        super().__init__()
        self.emb = get_emb()
        self.RNN_1 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.RNN_2 = nn.LSTM(hidden_dim * 4 * 2, hidden_dim, batch_first=True, bidirectional=True)

        self.theta2 = nn.Linear(hidden_dim * 2, 1, bias=False)

        # self.theta3 = nn.Bilinear(hidden_dim * 8, hidden_dim * 8, 1)
        self.theta3 = nn.Parameter(torch.zeros(hidden_dim * 2, hidden_dim * 2))
        bound = 1 / math.sqrt(hidden_dim * 2)
        init.uniform_(self.theta3, -bound, bound)

        self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim * 2)

        self.final = nn.Sequential(
            nn.Linear(4 * 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 if num_class == 2 else num_class),
        )

        self.T = 5

    def forward(self, x1, x2):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD

        xx1, xx2 = self.emb(x1), self.emb(x2)

        cs, _ = self.RNN_1(xx1)  # BL(2H)
        rs, _ = self.RNN_1(xx2)

        # cs = max_out(cs, 2) # BL(H)
        # rs = max_out(rs, 2)

        att = torch.einsum('bmh,bnh->bmn', [cs, rs])

        att_mask = c_mask.unsqueeze(2) | r_mask.unsqueeze(1)

        att = att.masked_fill(att_mask, -INF)

        c_att_scores = att.softmax(2)  # BMN
        c_hat = torch.einsum('bmn,bnh->bmh', [c_att_scores, rs])

        r_att_scores = att.softmax(1)  # BMN
        r_hat = torch.einsum('bmn,bmh->bnh', [r_att_scores, cs])

        c_m = torch.cat([cs, c_hat, cs - c_hat, cs * c_hat], -1)  # BL(4H)
        r_m = torch.cat([rs, r_hat, rs - r_hat, rs * r_hat], -1)

        c_m = c_m.masked_fill(c_mask.unsqueeze(-1), 0)
        r_m = r_m.masked_fill(r_mask.unsqueeze(-1), 0)

        c_v, _ = self.RNN_2(c_m)
        r_v, _ = self.RNN_2(r_m)

        c_v = c_v.masked_fill(c_mask.unsqueeze(-1), 0)
        r_v = r_v.masked_fill(r_mask.unsqueeze(-1), 0)

        x_t = torch.cat([c_v[:, -1, :]], -1)
        s_t = torch.cat([r_v[:, -1, :]], -1)

        p = []
        for t in range(self.T):
            p_t = self.final(torch.cat([s_t, x_t, s_t - x_t, s_t * x_t], -1)).squeeze(-1)
            p.append(p_t)

            if t != self.T - 1:
                beta = torch.einsum('bh,hj,bmj->bm', [s_t, self.theta3, c_v]).softmax(-1)
                x_t = torch.einsum('bm,bmh->bh', [beta, c_v])  # x_{t+1}
                s_t = self.gru(x_t, s_t)  # s_{t}

        p = torch.stack(p)
        p = p.mean(0)
        return p


def max_out(x, pool_size):
    return x.view(*x.shape[:-1], x.shape[-1] // pool_size, pool_size).max(-1)[0]


class SAN(nn.Module):

    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, num_class=NUM_CLASS, T=5):
        super().__init__()
        self.emb = get_emb()
        self.pwffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.RNN_1 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.RNN_2 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.RNN_3 = nn.LSTM(hidden_dim * 4, hidden_dim * 4, batch_first=True, bidirectional=True)

        self.pre_attention_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        self.gru = nn.GRUCell(hidden_dim * 8, hidden_dim * 8)

        self.theta2 = nn.Linear(hidden_dim * 8, 1, bias=False)

        # self.theta3 = nn.Bilinear(hidden_dim * 8, hidden_dim * 8, 1)
        self.theta3 = nn.Parameter(torch.zeros(hidden_dim * 8, hidden_dim * 8))
        bound = 1 / math.sqrt(hidden_dim * 8)
        init.uniform_(self.theta3, -bound, bound)

        self.theta4 = nn.Linear(hidden_dim * 4 * 8, 1 if num_class == 2 else num_class, bias=False)

        self.T = T
        # self.final = nn.Sequential(
        #     nn.Linear(2 * 2 * 4 * hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1 if num_class == 2 else num_class),
        # )

    def forward(self, x1, x2):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD

        xx1, xx2 = self.emb(x1), self.emb(x2)

        cs1, _ = self.RNN_1(xx1)  # BL(2H)
        rs1, _ = self.RNN_1(xx2)

        cs1 = max_out(cs1, 2)
        rs1 = max_out(rs1, 2)

        cs2, _ = self.RNN_2(cs1)
        rs2, _ = self.RNN_2(rs1)

        cs2 = max_out(cs2, 2)
        rs2 = max_out(rs2, 2)

        cs = torch.cat([cs1, cs2], -1)  # BL2H
        rs = torch.cat([rs1, rs2], -1)

        cs_ = F.relu(self.pre_attention_linear(cs))
        rs_ = F.relu(self.pre_attention_linear(rs))

        att = torch.einsum('bmh,bnh->bmn', [cs_, rs_])

        # att = F.dropout2d(att, 0, training=?)

        att_mask = c_mask.unsqueeze(2) | r_mask.unsqueeze(1)

        att = att.masked_fill(att_mask, -INF)

        c_att_scores = att.softmax(2)  # BMN
        c_hat = torch.einsum('bmn,bnh->bmh', [c_att_scores, rs])

        r_att_scores = att.softmax(1)  # BMN
        r_hat = torch.einsum('bmn,bmh->bnh', [r_att_scores, cs])

        c_u = torch.cat([cs, c_hat], -1)  # BL(4H)
        r_u = torch.cat([rs, r_hat], -1)

        c_m, _ = self.RNN_3(c_u)  # BL(8H)
        r_m, _ = self.RNN_3(r_u)

        # c_m = max_out(c_m, 2)
        # r_m = max_out(r_m, 2)

        # BL
        p = []
        s_t = torch.einsum('bmh,bm->bh', [r_m, self.theta2(r_m).squeeze(-1).softmax(-1)])  # Pooling the hypothesis

        for t in range(0, self.T):
            beta = torch.einsum('bh,hj,bmj->bm', [s_t, self.theta3, c_m]).softmax(-1)
            x_t = torch.einsum('bmh,bm->bh', [c_m, beta])
            p_t = self.theta4(torch.cat([x_t, s_t, x_t - s_t, x_t * s_t], -1)).squeeze(-1)
            p.append(p_t)
            if t != self.T - 1:
                s_t = self.gru(x_t, s_t)

        p = torch.stack(p)
        p = p.mean(0)

        # p = self.final(v).squeeze(-1)

        return p


class SAN_light(nn.Module):

    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, num_class=NUM_CLASS):
        super().__init__()
        self.emb = get_emb()

        self.RNN_1 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.RNN_2 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.RNN_3 = nn.LSTM(hidden_dim * 4, hidden_dim * 4, batch_first=True, bidirectional=True)

        self.gru = nn.GRUCell(hidden_dim * 4, hidden_dim * 4)

        self.theta2 = nn.Linear(hidden_dim * 4, 1, bias=False)

        # self.theta3 = nn.Bilinear(hidden_dim * 8, hidden_dim * 8, 1)
        self.theta3 = nn.Parameter(torch.zeros(hidden_dim * 8, hidden_dim * 8))
        bound = 1 / math.sqrt(hidden_dim * 4)
        init.uniform_(self.theta3, -bound, bound)

        self.theta4 = nn.Linear(hidden_dim * 4 * 4, 1 if NUM_CLASS == 2 else NUM_CLASS, bias=False)

        # self.final = nn.Sequential(
        #     nn.Linear(2 * 2 * 4 * hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1 if num_class == 2 else num_class),
        # )

        self.T = 5

    def forward(self, x1, x2):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD

        xx1, xx2 = self.emb(x1), self.emb(x2)

        cs1, _ = self.RNN_1(xx1)  # BL(2H)
        rs1, _ = self.RNN_1(xx2)

        cs1 = max_out(cs1, 2)
        rs1 = max_out(rs1, 2)

        cs2, _ = self.RNN_2(cs1)
        rs2, _ = self.RNN_2(rs1)

        cs2 = max_out(cs2, 2)
        rs2 = max_out(rs2, 2)

        cs = torch.cat([cs1, cs2], -1)  # BL2H
        rs = torch.cat([rs1, rs2], -1)

        att = torch.einsum('bmh,bnh->bmn', [cs, rs])

        # att = F.dropout2d(att, 0, training=?)

        att_mask = c_mask.unsqueeze(2) | r_mask.unsqueeze(1)

        att = att.masked_fill(att_mask, -INF)

        c_att_scores = att.softmax(2)  # BMN
        c_hat = torch.einsum('bmn,bnh->bmh', [c_att_scores, rs])

        r_att_scores = att.softmax(1)  # BMN
        r_hat = torch.einsum('bmn,bmh->bnh', [r_att_scores, cs])

        c_u = torch.cat([cs, c_hat], -1)  # BL(4H)
        r_u = torch.cat([rs, r_hat], -1)

        c_m, _ = self.RNN_3(c_u)  # BL(8H)
        r_m, _ = self.RNN_3(r_u)

        c_m = max_out(c_m, 2)
        r_m = max_out(r_m, 2)

        # BL
        p = []
        s_t = torch.einsum('bmh,bm->bh', [c_m, self.theta2(c_m).squeeze(-1).softmax(-1)])  # Pooling the premise

        for t in range(0, self.T):
            x_t = torch.einsum('bnh,bn->bh', [r_m, torch.einsum('bh,bnh->bn', [s_t, r_m]).softmax(-1)])
            p_t = self.theta4(torch.cat([x_t, s_t, x_t - s_t, x_t * s_t], -1)).squeeze(-1)
            p.append(p_t)
            if t != self.T - 1:
                s_t = self.gru(x_t, s_t)

        p = torch.stack(p)
        p = p.mean(0)

        # p = self.final(v).squeeze(-1)

        return p


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class RE2(nn.Module):
    class Block(nn.Module):

        def __init__(self, dim_in, dim_hidden, dropout=0.):
            super().__init__()
            # self.conv = nn.Conv1d(in_channels=dim_in, out_channels=dim, kernel_size=3, padding=3 // 2)
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(in_channels=dim_in if i == 0 else dim_hidden, out_channels=dim_hidden, kernel_size=3,
                              padding=3 // 2),
                    GeLU(),
                    nn.Dropout(dropout)
                )
                for i in range(3)])

            dim_both = dim_hidden + dim_in

            self.f = nn.Sequential(
                nn.Linear(dim_both, dim_both),
                GeLU()
            )
            # self.temperature = 1 / sqrt(dim_hidden)
            self.temperature = nn.Parameter(torch.tensor(1 / sqrt(dim_hidden)))

            self.g1 = nn.Sequential(
                nn.Linear(2 * dim_both, dim_hidden),
                GeLU(),
            )
            self.g2 = nn.Sequential(
                nn.Linear(2 * dim_both, dim_hidden),
                GeLU(),
            )
            self.g3 = nn.Sequential(
                nn.Linear(2 * dim_both, dim_hidden),
                GeLU(),
            )
            self.g = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(3 * dim_hidden, dim_hidden),
                GeLU(),
            )

        def encode(self, x, mask):
            xx = x.transpose(1, 2)
            for c in self.convs:
                xx = c(xx)
                xx = xx.masked_fill(mask.unsqueeze(-2), 0)
            xx = xx.transpose(1, 2)
            return xx

        def align(self, xx1, xx2, mask1, mask2):
            xx1_ = self.f(xx1)
            xx1_ = xx1_.masked_fill(mask1.unsqueeze(-1), 0)

            xx2_ = self.f(xx2)
            xx2_ = xx2_.masked_fill(mask2.unsqueeze(-1), 0)

            sims = torch.einsum('bme,bne->bmn', [xx1_, xx2_])  # * self.temperature
            mask = mask1.unsqueeze(2) | mask2.unsqueeze(1)
            sims = sims.masked_fill(mask, -INF)

            xx1_prim = torch.einsum('bne,bmn->bme', [xx2, sims.softmax(2)])
            xx2_prim = torch.einsum('bme,bmn->bne', [xx1, sims.softmax(1)])

            return xx1_prim, xx2_prim

        def fusion(self, xx, xx_prim, mask):
            xx_bar1 = self.g1(torch.cat([xx, xx_prim], -1))
            xx_bar2 = self.g2(torch.cat([xx, xx - xx_prim], -1))
            xx_bar3 = self.g3(torch.cat([xx, xx * xx_prim], -1))
            xx = self.g(torch.cat([xx_bar1, xx_bar2, xx_bar3], -1))
            xx = xx.masked_fill(mask.unsqueeze(-1), 0)
            return xx

        def forward(self, a, b, mask_a, mask_b):
            ae = self.encode(a, mask_a)
            be = self.encode(b, mask_b)

            aa = torch.cat([a, ae], -1)
            bb = torch.cat([b, be], -1)

            aa_prim, bb_prim = self.align(aa, bb, mask_a, mask_b)

            aa = self.fusion(aa, aa_prim, mask_a)
            bb = self.fusion(bb, bb_prim, mask_b)
            return aa, bb

    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0):
        super().__init__()
        self.emb = get_emb()
        self.blocks = nn.ModuleList([
            RE2.Block(emb_dim, hidden_dim, dropout=dropout),
            RE2.Block(emb_dim + hidden_dim, hidden_dim, dropout=dropout),
            RE2.Block(emb_dim + hidden_dim, hidden_dim, dropout=dropout)
        ])
        self.h = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 if NUM_CLASS == 2 else NUM_CLASS)
        )

    def forward(self, x1, x2):
        mask_a = x1 == PAD
        mask_b = x2 == PAD

        a_emb = self.emb(x1)
        b_emb = self.emb(x2)

        a = a_emb
        b = b_emb

        for i, block in enumerate(self.blocks):
            if i == 0:
                res_a, res_b = a, b
            elif i == 1:
                res_a, res_b = a, b
                a = torch.cat([a, a_emb], -1)
                b = torch.cat([b, b_emb], -1)
            else:
                res_a, res_b = a, b
                a = torch.cat([(a + res_a) * sqrt(.5), a_emb], -1)
                b = torch.cat([(b + res_b) * sqrt(.5), b_emb], -1)
            a, b = block(a, b, mask_a, mask_b)


        # a = a.masked_fill(mask_a.unsqueeze(-1), -INF)
        # b = b.masked_fill(mask_b.unsqueeze(-1), -INF)

        a = a.max(1)[0]
        b = b.max(1)[0]

        y_hat = self.h(torch.cat([a, b, a - b, a * b], -1)).squeeze(-1)

        return y_hat


class CustomREE(nn.Module):
    class Block(nn.Module):

        def __init__(self, dim_in, dim):
            super().__init__()
            self.conv = nn.Conv1d(in_channels=dim_in, out_channels=dim, kernel_size=3, padding=3 // 2)
            self.bilstm = nn.LSTM(dim_in, dim // 2, bidirectional=True)
            dim_ = dim * 2
            self.f = nn.Linear(dim_, dim_)
            self.rnn2 = nn.LSTM(4 * dim_, dim_ // 2, bidirectional=True)
            # self.g2 = nn.Linear(2 * dim_, dim_)
            # self.g3 = nn.Linear(2 * dim_, dim_)
            # self.g = nn.Linear(3 * dim_, dim)

        def encode(self, x, mask):
            xx = F.relu(self.conv(x.transpose(1, 2)).transpose(1, 2))
            xx = xx.masked_fill(mask.unsqueeze(-1), 0)
            return xx

        def encode2(self, x, mask):
            xx = self.bilstm(x)[0]
            # xx = xx.masked_fill(mask.unsqueeze(-1), 0)
            return xx

        def align(self, xx1, xx2, mask1, mask2):
            # xx1_ = F.relu(self.f(xx1))
            # xx1_ = xx1_.masked_fill(mask1.unsqueeze(-1), 0)

            # xx2_ = F.relu(self.f(xx2))
            # xx2_ = xx2_.masked_fill(mask2.unsqueeze(-1), 0)

            sims = torch.einsum('bme,bne->bmn', [xx1, xx2])
            mask = mask1.unsqueeze(2) | mask2.unsqueeze(1)
            sims = sims.masked_fill(mask, -INF)

            xx1_prim = torch.einsum('bne,bmn->bme', [xx2, sims.softmax(2)])
            xx2_prim = torch.einsum('bme,bmn->bne', [xx1, sims.softmax(1)])

            ## FIXME mask?

            return xx1_prim, xx2_prim

        def fusion(self, xx, xx_prim):
            xx_bar = self.rnn2(torch.cat([xx, xx_prim, xx - xx_prim, xx * xx_prim], -1))[0]
            return xx_bar

        def forward(self, ai, bi, mask_a, mask_b, ap=None, bp=None):
            if ap is not None:
                aa = torch.cat([ai, ap], -1)
                bb = torch.cat([bi, bp], -1)
            else:
                aa = ai
                bb = bi

            ae = self.encode(aa, mask_a)
            be = self.encode(bb, mask_b)

            if ap is not None:
                aa = torch.cat([ai, ae + ap], -1)
                bb = torch.cat([bi, be + bp], -1)
            else:
                aa = torch.cat([ai, ae], -1)
                bb = torch.cat([bi, be], -1)

            aa_prim, bb_prim = self.align(aa, bb, mask_a, mask_b)

            aa = self.fusion(aa, aa_prim)
            bb = self.fusion(bb, bb_prim)
            return aa, bb

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.block1 = RE2.Block(EMBEDDING_DIM, EMBEDDING_DIM)
        self.block2 = RE2.Block(2 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.block3 = RE2.Block(2 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.h = nn.Sequential(
            nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, 1)
        )

    def forward(self, x1, x2):
        mask_a = x1 == PAD
        mask_b = x2 == PAD

        a_emb = self.emb(x1)
        b_emb = self.emb(x2)

        a_out1, b_out1 = self.block1(a_emb, b_emb, mask_a, mask_b)
        a_out2, b_out2 = self.block2(a_emb, b_emb, mask_a, mask_b, a_out1, b_out1)
        a_out3, b_out3 = self.block3(a_emb, b_emb, mask_a, mask_b, a_out2, b_out2)

        a_emb = a_out3.max(1)[0]
        b_emb = b_out3.max(1)[0]

        y_hat = self.h(torch.cat([a_emb, b_emb, a_emb - b_emb, a_emb * b_emb], -1)).squeeze(-1)

        return y_hat


class CompAgg(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.preprocess = nn.LSTM(EMBEDDING_DIM, EMBEDDING_DIM, bidirectional=True)
        self.wg = nn.Linear(2 * EMBEDDING_DIM, 2 * EMBEDDING_DIM)
        self.w = nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM * 4)
        self.agg = nn.Conv1d(in_channels=EMBEDDING_DIM * 4, out_channels=EMBEDDING_DIM * 4,
                             kernel_size=3, padding=3 // 2)
        self.final = nn.Sequential(
            nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, 1)
        )

    def forward(self, x1, x2):
        q, a = self.emb(x1), self.emb(x2)
        q_bar, a_bar = self.preprocess(q)[0], self.preprocess(a)[0]
        g = torch.einsum('bme,bne->bmn', [self.wg(q_bar), a_bar]).softmax(-1)
        h = torch.einsum('bme,bmn->bne', [q_bar, g])

        t = self.w(torch.cat([(a_bar - h) * (a_bar - h), a_bar * h], -1)).relu()  # BN(2E)

        o = self.agg(t.transpose(1, 2)).transpose(1, 2)  # BN(2E)

        o = o.max(1)[0]

        o = self.final(o).squeeze()

        return o


class BiMPM(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.context = nn.LSTM(EMBEDDING_DIM, EMBEDDING_DIM, bidirectional=True)
        perspective = 50
        self.w1 = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(perspective, EMBEDDING_DIM * 2)))
        self.w2 = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(perspective, EMBEDDING_DIM * 2)))
        self.w3 = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(perspective, EMBEDDING_DIM * 2)))
        self.w4 = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(perspective, EMBEDDING_DIM * 2)))

        self.agg = nn.LSTM(perspective, perspective, bidirectional=True)
        self.agg2 = nn.LSTM(2 * EMBEDDING_DIM, 2 * EMBEDDING_DIM, bidirectional=True)
        self.final = nn.Sequential(
            nn.Linear(4 * EMBEDDING_DIM * 2 * 4, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, 1)
        )

    def forward(self, x1, x2):
        p, q = self.emb(x1), self.emb(x2)
        hp, hq = self.context(p)[0], self.context(q)[0]

        # bmnl = self.multi_perspective_cosine_many2many(hp, hq, self.w1)
        # full_matching_p = bmnl[:, :, -1, :]
        # full_matching_q = bmnl[:, -1, :, :]
        #
        # max_pool_matching_p = bmnl.max(2)[0]
        # max_pool_matching_q = bmnl.max(1)[0]

        bmn = self.cosine(hp, hq)
        h_mean_p = torch.einsum('bmn,bne->bme', [bmn.softmax(2), hq])
        h_mean_q = torch.einsum('bmn,bme->bne', [bmn.softmax(1), hp])
        # attentive_matching_p = self.multi_perspective_cosine_one2one(hp, h_mean_p, self.w1)
        # attentive_matching_q = self.multi_perspective_cosine_one2one(hp, h_mean_q, self.w2)

        # h_max = hq.max(1)[0]
        # max_matching = self.multi_perspective_cosine_many2one(hp, h_max, self.w3)

        matching_p = torch.cat([
            self.agg2(hp)[0][:, -1, :],
            self.agg2(h_mean_p)[0][:, -1, :],
            # self.agg(full_matching_p)[0][:, -1, :],
            # self.agg(max_pool_matching_p)[0][:, -1, :],
        ], -1)

        matching_q = torch.cat([
            self.agg2(hq)[0][:, -1, :],  # 4E
            self.agg2(h_mean_q)[0][:, -1, :],  # 4E
            # self.agg(full_matching_q)[0][:, -1, :],  # 2p
            # self.agg(max_pool_matching_q)[0][:, -1, :],
        ], -1)

        matching = torch.cat([
            matching_p,
            matching_q,
            matching_p - matching_q,
            matching_p * matching_q
        ], -1)

        return self.final(matching).squeeze(-1)

    @staticmethod
    def multi_perspective_cosine_many2many(m, n, w):
        """

        :param m: BMD
        :param n: BND
        :param w: LD
        :return: BMNL
        """
        m = torch.einsum('bmd,ld->bmld', [m, w])
        n = torch.einsum('bnd,ld->bnld', [n, w])
        cosine_sim_numerator = torch.einsum('bmld,bnld->bmnl', [m, n])
        cosine_sim_denominator = torch.einsum('bml,bnl->bmnl', [m.norm(dim=-1), n.norm(dim=-1)])
        cosine_sim_denominator[cosine_sim_denominator == 0] = EPSILON
        cosine_sim = cosine_sim_numerator / cosine_sim_denominator
        return cosine_sim

    @staticmethod
    def multi_perspective_cosine_one2one(m1, m2, w):
        """

        :param m1: BMD
        :param m2: BMD
        :param w: LD
        :return: BML
        """
        m1 = torch.einsum('bmd,ld->bmld', [m1, w])
        m2 = torch.einsum('bmd,ld->bmld', [m2, w])
        cosine_sim_numerator = torch.einsum('bmld,bmld->bml', [m1, m2])
        cosine_sim_denominator = torch.einsum('bml,bml->bml', [m1.norm(dim=-1), m2.norm(dim=-1)])
        cosine_sim_denominator[cosine_sim_denominator == 0] = EPSILON
        cosine_sim = cosine_sim_numerator / cosine_sim_denominator
        return cosine_sim

    @staticmethod
    def multi_perspective_cosine_many2one(m1, m2, w):
        """

        :param m1: BMD
        :param m2: BD
        :param w: LD
        :return: BML
        """
        m1 = torch.einsum('bmd,ld->bmld', [m1, w])
        m2 = torch.einsum('bd,ld->bld', [m2, w])
        cosine_sim_numerator = torch.einsum('bmld,bld->bml', [m1, m2])
        cosine_sim_denominator = torch.einsum('bml,bl->bml', [m1.norm(dim=-1), m2.norm(dim=-1)])
        cosine_sim_denominator[cosine_sim_denominator == 0] = EPSILON
        cosine_sim = cosine_sim_numerator / cosine_sim_denominator
        return cosine_sim

    @staticmethod
    def cosine(m, n) -> torch.Tensor:
        """
        :param m: BMD
        :param n: BND
        :return: BMN
        """
        cosine_sim_numerator = torch.einsum('bmd,bnd->bmn', [m, n])
        cosine_sim_denominator = torch.einsum('bm,bn->bmn', [m.norm(dim=-1), n.norm(dim=-1)])
        cosine_sim_denominator[cosine_sim_denominator == 0] = EPSILON
        cosine_sim = cosine_sim_numerator / cosine_sim_denominator
        return cosine_sim


class FlatSMN(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.gru = nn.GRU(EMBEDDING_DIM, EMBEDDING_DIM, bidirectional=True)
        self.cnn = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.w = nn.Parameter(nn.init.kaiming_normal_(torch.zeros(2 * EMBEDDING_DIM, 2 * EMBEDDING_DIM)))
        o_dim = 8 * (MAX_LEN // 3) ** 2
        self.final = nn.Sequential(
            nn.Linear(o_dim, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM, 1)
        )

    def forward(self, x1, x2):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD
        xx1, xx2 = self.emb(x1), self.emb(x2)
        xx1_gru, xx2_gru = self.gru(xx1)[0], self.gru(xx2)[0]
        m1 = torch.einsum('bme,bne->bmn', [xx1, xx2]).relu()
        m2 = torch.einsum('bme,ef,bnf->bmn', [xx1_gru, self.w, xx2_gru]).relu()
        m = torch.stack([m1, m2], 1)

        mask = (c_mask.unsqueeze(2) | r_mask.unsqueeze(1)).unsqueeze(1)

        m = m.masked_fill(mask, 0)

        o = self.cnn(m).relu()
        o = F.max_pool2d(o, 3)
        o = o.view(m.shape[0], -1)

        return self.final(o).squeeze(-1)


class RNN(nn.Module):

    def __init__(self, dim=100, hidden_dim=200):
        super().__init__()
        self.emb = get_emb()
        self.rnn = nn.LSTM(dim, hidden_dim, batch_first=True, bidirectional=True)
        self.final = nn.Linear(hidden_dim * 2 * 2 * 5, 1 if NUM_CLASS == 2 else NUM_CLASS)

    def forward(self, x1, x2):
        x1_mask = x1 == PAD
        x2_mask = x2 == PAD

        xx1 = self.emb(x1)
        xx2 = self.emb(x2)
        xx1 = self.rnn(xx1)[0]
        xx2 = self.rnn(xx2)[0]

        xx1 = xx1.masked_fill(x1_mask.unsqueeze(-1), -INF)
        xx2 = xx2.masked_fill(x2_mask.unsqueeze(-1), -INF)

        xx1_max = xx1.max(1)[0]
        xx2_max = xx2.max(1)[0]

        xx1 = xx1.masked_fill(x1_mask.unsqueeze(-1), 0)
        xx2 = xx2.masked_fill(x2_mask.unsqueeze(-1), 0)

        xx1_mean = xx1.sum(1) / (~x1_mask).long().sum(1, keepdim=True).float()
        xx2_mean = xx2.sum(1) / (~x2_mask).long().sum(1, keepdim=True).float()

        xx1 = torch.cat([xx1_max, xx1_mean], -1)
        xx2 = torch.cat([xx2_max, xx2_mean], -1)

        m = torch.cat([xx1, xx2, xx1 - xx2, xx1 * xx2, (xx1 - xx2) * (xx2 - xx1)], -1)
        y_hat = self.final(m).squeeze(-1)

        return y_hat


class DualTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.rnn = nn.Transformer(EMBEDDING_DIM, )
        self.final = nn.Linear(400 * 4, 1 if NUM_CLASS == 2 else NUM_CLASS)

    def forward(self, x1, x2):
        xx1 = self.emb(x1)
        xx2 = self.emb(x2)
        xx1 = self.rnn(xx1)[0][:, -1, :]
        xx2 = self.rnn(xx2)[0][:, -1, :]

        m = torch.cat([xx1, xx2, xx1 - xx2, xx1 * xx2], -1)
        y_hat = self.final(m)

        return y_hat


class CAFE(nn.Module):
    class HighwayMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.H = nn.Linear(dim, dim)
            self.T = nn.Linear(dim, dim)

        def forward(self, x):
            t = self.T(x).sigmoid()
            return self.H(x).relu() * t + x * (1 - t)

    # class Factor(nn.Module):
    #
    #     def __init__(self, dim) -> None:
    #         super().__init__()
    #         self.linear = nn.Linear(dim, 1)
    #         self.V = nn.Parameter(torch.zeros(dim, dim // 2))
    #         nn.init.normal_(self.V)
    #
    #     def forward(self, xx):
    #         a = self.linear(xx).squeeze(-1)
    #         b = torch.triu(
    #             ((self.V @ self.V.t()).unsqueeze(0).unsqueeze(0) + (xx.unsqueeze(-2) + xx.unsqueeze(-1)))
    #         ).sum(-1).sum(-1)
    #         return a + b

    class Factor(nn.Module):
        def __init__(self, dim, only_linear=False):
            super().__init__()
            self.only_linear = only_linear
            k = dim
            # Initially we fill V with random values sampled from Gaussian distribution
            # NB: use nn.Parameter to compute gradients
            self.V = nn.Parameter(torch.randn(dim, k), requires_grad=True)
            nn.init.kaiming_normal_(self.V)
            self.lin = nn.Linear(dim, 1)

        def forward(self, x):
            if not self.only_linear:
                out_1 = (x @ self.V).pow(2).sum(-1)  # S_1^2
                out_2 = (x.pow(2) @ self.V.pow(2)).sum(-1)  # S_2
                out_inter =  0.5 * (out_1 - out_2)
            else:
               out_inter = 0
            out_lin = self.lin(x).squeeze(-1)
            out = out_inter + out_lin

            return out

    def __init__(self, dim=EMBEDDING_DIM) -> None:
        super().__init__()
        self.emb = get_emb()
        self.highways = nn.Sequential(*[self.HighwayMLP(dim) for _ in range(1)])
        self.F = nn.Linear(dim, dim)
        self.G = nn.Linear(dim, dim)
        self.Z = self.Factor(dim)
        self.Z2 = self.Factor(dim * 2)
        self.rnn_1 = nn.LSTM(dim, dim , bidirectional=False, batch_first=True)
        self.rnn_2 = nn.LSTM(dim + 6, dim, bidirectional=True, batch_first=True)
        self.final = nn.Sequential(
            nn.Linear(dim * 2 * 2 * 5, dim),
            nn.ReLU(),
            nn.Linear(dim, 1 if NUM_CLASS == 2 else NUM_CLASS)
        )

    def encode(self, xx):
        xx = self.emb(xx)
        xx = self.highways(xx)
        return xx

    def intra_align(self, xx, mask):
        xx_f = self.G(xx).relu()
        f = torch.einsum('bme,bne->bmn', [xx_f, xx_f])
        mask_2d = mask.unsqueeze(-1) | mask.unsqueeze(-2) | torch.eye(xx.shape[1]).bool().to(device)
        f = f.masked_fill(mask_2d, -INF)
        xx_hat = torch.einsum('bne,bmn->bme', [xx, f.softmax(-1)])
        return xx_hat

    def forward(self, x1, x2):
        c, r = x1, x2
        c_mask = c == PAD
        r_mask = r == PAD
        mask_2d = c_mask.unsqueeze(-1) | r_mask.unsqueeze(-2)

        c, r = self.emb(c), self.emb(r)
        # c, r = self.highways(c), self.highways(r)
        # c, r = self.rnn_1(c)[0], self.rnn_1(r)[0]

        c = c.masked_fill(c_mask.unsqueeze(-1), 0)
        r = r.masked_fill(r_mask.unsqueeze(-1), 0)

        c_f, r_f = self.F(c).relu(), self.F(r).relu()
        e = torch.einsum('bme,bne->bmn', [c_f, r_f])  # * (1/math.sqrt(EMBEDDING_DIM))
        e = e.masked_fill(mask_2d, -INF)
        c_hat = torch.einsum('bne,bmn->bme', [r, e.softmax(-1)])
        r_hat = torch.einsum('bme,bmn->bne', [c, e.softmax(-2)])

        c_intra = self.intra_align(c, c_mask)
        r_intra = self.intra_align(r, r_mask)

        c_hat = c_hat.masked_fill(c_mask.unsqueeze(-1), 0)
        r_hat = r_hat.masked_fill(r_mask.unsqueeze(-1), 0)
        c_intra = c_intra.masked_fill(c_mask.unsqueeze(-1), 0)
        r_intra = r_intra.masked_fill(r_mask.unsqueeze(-1), 0)

        # self.Z(torch.cat([c_hat, r_hat], -1))
        factorizations_c = torch.stack([
            self.Z2(torch.cat([c_hat, c], -1)),
            self.Z(c_hat - c),
            self.Z(c_hat * c),
            self.Z2(torch.cat([c_intra, c], -1)),
            self.Z(c_intra - c),
            self.Z(c_intra * c),

        ], -1)

        factorizations_r = torch.stack([
            self.Z2(torch.cat([r_hat, r], -1)),
            self.Z(r_hat - r),
            self.Z(r_hat * r),
            self.Z2(torch.cat([r_intra, r], -1)),
            self.Z(r_intra - r),
            self.Z(r_intra * r),

        ], -1)

        c_x = torch.cat([c, factorizations_c], -1)
        r_x = torch.cat([r, factorizations_r], -1)

        c_x = c_x.masked_fill(c_mask.unsqueeze(-1), 0)
        r_x = r_x.masked_fill(r_mask.unsqueeze(-1), 0)

        c_x = self.rnn_2(c_x)[0]
        r_x = self.rnn_2(r_x)[0]

        c_x = c_x.masked_fill(c_mask.unsqueeze(-1), -INF)
        r_x = r_x.masked_fill(r_mask.unsqueeze(-1), -INF)

        c_x_max = c_x.max(1)[0]
        r_x_max = r_x.max(1)[0]

        c_x = c_x.masked_fill(c_mask.unsqueeze(-1), 0)
        r_x = r_x.masked_fill(r_mask.unsqueeze(-1), 0)

        c_x_mean = c_x.sum(1) / (~c_mask).sum(1, keepdim=True).float()
        r_x_mean = r_x.sum(1) / (~r_mask).sum(1, keepdim=True).float()

        c_p = torch.cat([c_x_max, c_x_mean], -1)
        r_p = torch.cat([r_x_max, r_x_mean], -1)

        penult = torch.cat([c_p, r_p, c_p - r_p, c_p * r_p, (c_p - r_p) * (r_p - c_p)], -1)
        out = self.final(penult).squeeze(-1)

        return out


# %%
# model = RE2(dropout=DROPOUT)
# model = CustomREE()
# model = RNN()
model = ESIM_()
# model = CompAgg()
# model = DiSAN()
# model = BiMPM()
# model = FlatSMN()
# model = SAN(T=5)
# model = SAN_light()
# model = ESIM_SAN()
# model = CAFE()
# model = TEMP()

model = model.to(device)

print(type(model).__name__)


# %%
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


# def dont_update_unkown_words(model):
#     indices = torch.LongTensor([UNK]).to(device)
#     model.emb.weight.grad[indices] = 0

metrics_history = []
dev_metrics_history = []

for i_epoch in range(1, N_EPOCH + 1):
    model.train()
    loss_total = 0
    # accu_total = 0
    total = 0
    train_confusion_table = np.zeros((NUM_CLASS, NUM_CLASS))

    progress_bar = tqdm(train_data_loader)
    for batch in progress_bar:
        # for x1, x2, y in train_data_loader:
        optimizer.zero_grad()
        # weights = train_weights[y]
        batch = tuple(i.to(device) for i in batch)
        x1, x2, y = batch
        y_hat = model(x1, x2)

        if NUM_CLASS == 2:
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction='sum')
        else:
            loss = F.cross_entropy(y_hat, y, reduction='sum')

        # dont_update_unkown_words(model)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        assert grad_norm >= 0, 'encounter nan in gradients.'
        optimizer.step()

        loss_total += loss.item()
        total += y.shape[0]
        train_confusion_table += calc_confusion_matrix(y, y_hat, NUM_CLASS)

        metrics = (
            loss_total / total,
            *calc_metrics(train_confusion_table),
        )

        progress_bar.set_description(
            Fore.RESET +
            '[ EP {0:02d} ]'
            # '[ TRN LS: {1:.4f} PR: {2:.4f} F1: {4:.4f} AC: {5:.4f}]'
            '[ LS: {1:.4f} AC: {5:.4f} TRN ]'
            .format(i_epoch, *metrics))

    model.eval()
    loss_total_dev = 0
    total_dev = 0
    dev_confusion_table = np.zeros((NUM_CLASS, NUM_CLASS))

    with torch.no_grad():
        progress_bar = tqdm(dev_data_loader)
        for x1, x2, y in progress_bar:
            # for x1, x2, y in dev_data_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_hat = model(x1, x2)

            if NUM_CLASS == 2:
                loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), reduction='sum')
            else:
                loss = F.cross_entropy(y_hat, y, reduction='sum')

            loss_total_dev += loss.item()
            total_dev += y.shape[0]
            dev_confusion_table += calc_confusion_matrix(y, y_hat, NUM_CLASS)

            metrics_dev = (
                loss_total_dev / total_dev,
                *calc_metrics(dev_confusion_table)
            )

            progress_bar.set_description(
                Fore.GREEN +
                '[ EP {0:02d} ]'
                # '[ TST LS: {1:.4f} PR: {2:.4f} F1: {4:.4f} AC: {5:.4f} ]'
                '[ LS: {1:.4f} AC: {5:.4f} DEV ]'
                .format(i_epoch, *metrics_dev))

    metrics_history.append((*metrics, *metrics_dev))

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

# %%
hyper_parameters = '--'.join([k + '=' + str(v) for k, v in vars(args[0]).items() if k in ['embedding_dim', 'batch', 'dropout']])
history_dir = 'histories/' + hyper_parameters
if os.path.exists(history_dir + '/' + DATASET_NAME):
    with open(history_dir + '/' + DATASET_NAME, 'rb') as f:
        baselines = pickle.load(f)
else:
    baselines = dict()


def model_name(model):
    try:
        a = model.name
    except:
        a = type(model).__name__
    return a


baselines[model_name(model)] = metrics_history

metrics_history_all_t = [np.array(i).T for i in baselines.values()]
model_names = list(baselines.keys())

rows = []
dev_acc_index = -1
for name, values in zip(model_names, metrics_history_all_t):
    best_index = values[dev_acc_index].argmax()
    row = [name, *values[:, best_index].tolist(), best_index]
    row = row[:2] + row[5:7] + row[10:]
    rows.append(row)

rows = sorted(rows, key=lambda x: x[-2])
t = Texttable()
t.add_rows(rows, header=False)
t.header(('Model', 'Train Loss', 'Train Acc', 'Dev Loss', 'Dev Acc', 'Epoch'))
print(t.draw())

# %%
metrics_dir = 'metrics/' + hyper_parameters
if not os.path.exists(metrics_dir):
    os.mkdir(metrics_dir)
with open(metrics_dir + '/' + DATASET_NAME + '.txt', 'w') as f:
    print(t.draw(), file=f)

# %%

if not os.path.exists(history_dir):
    os.mkdir(history_dir)
with open(history_dir + '/' + DATASET_NAME, 'wb') as f:
    pickle.dump(baselines, f)

# %%
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()
plt.clf()
ax = plt.gca()
for history in metrics_history_all_t:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(range(1, len(history[dev_acc_index]) + 1), history[dev_acc_index], color=color, marker='o')
    # plt.plot(history[3], '--', color=color)

# plt.yticks([i / 10 for i in range(11)])
plt.ylabel('Acc')
plt.xlabel('Epoch')

legends = []
for i in model_names:
    legends.append(i)

plt.legend(legends, loc='upper left',
           bbox_to_anchor=(0, -0.2),
           fancybox=True, shadow=True, ncol=2)
plt.title("Comparison of ANN Models for " + DATASET_NAME)
# plt.show()
plot_dir = 'plots/' + hyper_parameters
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
plt.savefig(plot_dir + '/' + DATASET_NAME + '.png', dpi=400, bbox_inches='tight')
