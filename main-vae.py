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
from torch.nn import functional as F
from tqdm import tqdm

# %%
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# %%

tokenize = lambda x: x.split()
EMBEDDING_DIM = 100
PAD = 1
UNK = 0
device = torch.device('cuda')
EPSILON = 1e-13
INF = 1e13
PAD_FIRST = False
USE_EMB = False

# N_EPOCH = 10
# BATCH_SIZE = 32
# MAX_LEN = 25
# FIX_LEN = True
# DATASET_NAME = 'quora'
# NUM_CLASS = 2

# N_EPOCH = 10
# BATCH_SIZE = 32
# MAX_LEN = 32
# FIX_LEN = False
# DATASET_NAME = 'SciTailV1.1'
# NUM_CLASS = 2
# KEEP = 1000
# MIN_FREQ = 2

N_EPOCH = 10
BATCH_SIZE = 32
MAX_LEN = 30
FIX_LEN = False
DATASET_NAME = 'snli'
NUM_CLASS = 3
KEEP = 28_000
MIN_FREQ = 8

special_tokens = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
UNK, PAD, SOS, EOS = 0, 1, 2, 3


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
    words_set = [k for k, _ in texts_counter.most_common()]
    itos = special_tokens + words_set
    stoi = {v: i for i, v in enumerate(itos)}
    p_idx = df.iloc[:, 0].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    q_idx = df.iloc[:, 1].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    p_idx_sos = df.iloc[:, 0].apply(lambda x: ([SOS] + [stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    q_idx_sos = df.iloc[:, 1].apply(lambda x: ([SOS] + [stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    p_idx_eos = df.iloc[:, 0].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN] + [EOS]))
    q_idx_eos = df.iloc[:, 1].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN] + [EOS]))
    labels = df.iloc[:, 2]
    return stoi, itos, p_idx, q_idx, p_idx_sos, q_idx_sos, p_idx_eos, q_idx_eos, labels


def read_eval(fname, stoi):
    df = pd.read_csv(fname)
    df.iloc[:, 0] = df.iloc[:, 0].apply(tokenize)
    df.iloc[:, 1] = df.iloc[:, 1].apply(tokenize)
    p_idx = df.iloc[:, 0].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    q_idx = df.iloc[:, 1].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    p_idx_sos = df.iloc[:, 0].apply(lambda x: ([SOS] + [stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    q_idx_sos = df.iloc[:, 1].apply(lambda x: ([SOS] + [stoi[i] if i in stoi else UNK for i in x][:MAX_LEN]))
    p_idx_eos = df.iloc[:, 0].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN] + [EOS]))
    q_idx_eos = df.iloc[:, 1].apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x][:MAX_LEN] + [EOS]))
    labels = df.iloc[:, 2]
    return p_idx, q_idx, p_idx_sos, q_idx_sos, p_idx_eos, q_idx_eos, labels


class CustomDataset(Dataset):

    def __init__(self, p_idx, q_idx, p_idx_sos, q_idx_sos, p_idx_eos, q_idx_eos, labels, keep) -> None:
        super().__init__()
        self.ps = p_idx
        self.qs = q_idx
        self.ps_sos = p_idx_sos
        self.qs_sos = q_idx_sos
        self.ps_eos = p_idx_eos
        self.qs_eos = q_idx_eos
        self.labels = labels

        if keep is not None:
            self.labels[self.labels.sample(len(self.labels) - keep).index] = -1

    def __getitem__(self, index: int):
        return self.ps[index], self.qs[index], \
               self.ps_sos[index], self.qs_sos[index], \
               self.ps_eos[index], self.qs_eos[index], \
               self.labels[index]

    def __len__(self) -> int:
        return self.labels.__len__()


print('Reading Dataset {} ...'.format(DATASET_NAME))
stoi, itos, p_idx, q_idx, p_idx_sos, q_idx_sos, p_idx_eos, q_idx_eos, labels = read_train(DATASET_NAME + '/train.csv')
train_dataset = CustomDataset(p_idx, q_idx, p_idx_sos, q_idx_sos, p_idx_eos, q_idx_eos, labels, keep=KEEP)
dev_dataset = CustomDataset(*read_eval(DATASET_NAME + '/dev.csv', stoi), keep=None)
test_dataset = CustomDataset(*read_eval(DATASET_NAME + '/test.csv', stoi), keep=None)


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
    contexts_sos = torch.LongTensor([pad(item[2], m + 1) for item in batch])
    response_sos = torch.LongTensor([pad(item[3], n + 1) for item in batch])
    contexts_eos = torch.LongTensor([pad(item[4], m + 1) for item in batch])
    response_eos = torch.LongTensor([pad(item[5], n + 1) for item in batch])
    labels = torch.LongTensor([item[6] for item in batch])
    return contexts, response, contexts_sos, response_sos, contexts_eos, response_eos, labels


train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
dev_data_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

VOCAB_LEN = len(itos)
print('Vocab length is {}'.format(VOCAB_LEN))

# %%
def get_emb():
    if USE_EMB:
        print('Reading Embeddings...')
        w2v = gensim.models.KeyedVectors.load_word2vec_format(
            '/home/amir/IIS/Datasets/embeddings/glove.6B.' + str(EMBEDDING_DIM)
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
        emb = nn.Embedding(VOCAB_LEN, EMBEDDING_DIM, padding_idx=PAD, _weight=embedding_weights.clone())
        emb.weight.requires_grad = False
        return emb
    else:
        return nn.Embedding(VOCAB_LEN, EMBEDDING_DIM, padding_idx=PAD)


# %%

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#%%
class ESIM(nn.Module):

    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, num_class=NUM_CLASS):
        super().__init__()
        self.emb = get_emb()
        self.RNN_1 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.RNN_2 = nn.LSTM(hidden_dim * 2 * 4, hidden_dim, batch_first=True, bidirectional=True)
        self.final = nn.Sequential(
            nn.Linear(2 * 2 * 4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 if num_class == 2 else num_class),
        )

    def forward(self, x1, x2, _, __):
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

        c_m = torch.cat([cs, c_hat, cs - c_hat, cs * c_hat], -1)  # BL(8H)
        r_m = torch.cat([rs, r_hat, rs - r_hat, rs * r_hat], -1)

        c_v, _ = self.RNN_2(c_m)  # BL(16H)
        r_v, _ = self.RNN_2(r_m)

        # v = torch.cat([c_v[:, -1, :], r_v[:, -1, :], c_v.max(1)[0], r_v.max(1)[0],
        #                ], -1)

        vc = torch.cat([c_v[:, -1, :], c_v.max(1)[0]], -1)
        vr = torch.cat([r_v[:, -1, :], r_v.max(1)[0]], -1)
        v = torch.cat([vc, vr, vc - vr, vc * vr], -1)

        p = self.final(v).squeeze(-1)

        return p, torch.randn(*x1.shape, VOCAB_LEN), torch.randn(*x1.shape, VOCAB_LEN)


class ESIM_VAE(nn.Module):

    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, num_class=NUM_CLASS):
        super().__init__()
        self.emb = get_emb()
        self.RNN_1 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.RNN_2 = nn.LSTM(hidden_dim * 2 * 4, hidden_dim, batch_first=True, bidirectional=True)
        self.final = nn.Sequential(
            nn.Linear(2 * 2 * 4 * hidden_dim + 2 * 4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 if num_class == 2 else num_class),
        )

        self.context2decoder = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim * 2, batch_first=True)
        self.out2vocab = nn.Linear(hidden_dim * 2, VOCAB_LEN)

    def forward(self, x1, x2, x1_sos, x2_sos):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD

        xx1, xx2 = self.emb(x1), self.emb(x2)

        cs, _ = self.RNN_1(xx1)  # BL(2H)
        rs, _ = self.RNN_1(xx2)

        ## INTERACTION

        c_hat, r_hat = self.align(c_mask, cs, r_mask, rs)

        c_v = self.fuse(c_hat, cs)
        r_v = self.fuse(r_hat, rs)

        v = self.pool(c_v, r_v)



        ## DECODER

        xx1_decoder_input = self.emb(x1_sos)
        logits1, z1 = self.decode(xx1_decoder_input, cs[:, -1, :])

        xx2_decoder_input = self.emb(x2_sos)
        logits2, z2 = self.decode(xx2_decoder_input, rs[:, -1, :])

        pz = torch.cat([z1, z2, z1 - z2, z1 * z2, v], -1)

        p = self.final(pz).squeeze(-1)

        return p, logits1, logits2

    def decode(self, decoder_input, context):
        xx1_context = self.context2decoder(context).unsqueeze(0)
        xx1_hiddens, _ = self.decoder_rnn(decoder_input, (xx1_context, torch.zeros_like(xx1_context)))
        logits = self.out2vocab(xx1_hiddens)
        return logits, xx1_context.squeeze(0)

    def pool(self, c_v, r_v):
        # v = torch.cat([c_v[:, -1, :], r_v[:, -1, :], c_v.max(1)[0], r_v.max(1)[0],
        #                ], -1)

        vc = torch.cat([c_v[:, -1, :], c_v.max(1)[0]], -1)
        vr = torch.cat([r_v[:, -1, :], r_v.max(1)[0]], -1)
        v = torch.cat([vc, vr, vc - vr, vc * vr], -1)
        return v

    def fuse(self, x_hat, xs):
        x_m = torch.cat([xs, x_hat, xs - x_hat, xs * x_hat], -1)  # BL(8H)
        x_v, _ = self.RNN_2(x_m)
        return x_v

    def align(self, c_mask, cs, r_mask, rs):
        att = torch.einsum('bmh,bnh->bmn', [cs, rs])
        att_mask = c_mask.unsqueeze(2) | r_mask.unsqueeze(1)
        att = att.masked_fill(att_mask, -INF)
        c_att_scores = att.softmax(2)  # BMN
        c_hat = torch.einsum('bmn,bnh->bmh', [c_att_scores, rs])
        r_att_scores = att.softmax(1)  # BMN
        r_hat = torch.einsum('bmn,bmh->bnh', [r_att_scores, cs])
        return c_hat, r_hat


class LSTM_AE(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.encoder_rnn = nn.LSTM(EMBEDDING_DIM, 100, batch_first=True, bidirectional=True)
        self.decoder_rnn = nn.LSTM(EMBEDDING_DIM, 200, batch_first=True)
        self.fc2vocab = nn.Linear(200, VOCAB_LEN)
        self.latent_dim = 200
        self.match = nn.Linear(self.latent_dim * 4, 1 if NUM_CLASS == 2 else NUM_CLASS)

        # self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(200, VOCAB_LEN, [100, 1000, 5000])

    def forward(self, x1, x2, x1_sos, x2_sos):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD

        xx1, xx2 = self.emb(x1), self.emb(x2)

        xx1, _ = self.encoder_rnn(xx1)
        xx2, _ = self.encoder_rnn(xx2)

        z1 = xx1[:, -1, :]
        z2 = xx2[:, -1, :]

        xx1_sos = self.emb(x1_sos)
        xx2_sos = self.emb(x2_sos)

        xx1_hat, _ = self.decoder_rnn(xx1_sos, (z1.unsqueeze(0), torch.zeros_like(z1.unsqueeze(0))))
        xx2_hat, _ = self.decoder_rnn(xx2_sos, (z2.unsqueeze(0), torch.zeros_like(z2.unsqueeze(0))))

        logits1 = self.fc2vocab(xx1_hat)
        logits2 = self.fc2vocab(xx2_hat)

        m = torch.cat([z1, z2, z1 - z2, z1 * z2], -1)
        y_hat = self.match(m).squeeze(-1)

        return y_hat, logits1, logits2


class LSTM_VAE(nn.Module):

    def __init__(self, hidden, latent):
        super().__init__()
        self.emb = get_emb()
        self.encoder_rnn = nn.LSTM(EMBEDDING_DIM, hidden, batch_first=True, bidirectional=True)
        self.decoder_rnn = nn.LSTM(EMBEDDING_DIM, hidden * 2, batch_first=True)
        self.fc2vocab = nn.Linear(hidden * 2, VOCAB_LEN)
        self.latent_dim = latent
        self.match = nn.Linear(self.latent_dim * 4, 1 if NUM_CLASS == 2 else NUM_CLASS)
        self.context2mu = nn.Linear(hidden * 2, self.latent_dim)
        self.context2logvar = nn.Linear(hidden * 2, self.latent_dim)
        self.latent2context = nn.Linear(self.latent_dim, hidden * 2)
        # self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(200, VOCAB_LEN, [100, 1000, 5000])


    def reparam(self, context):
        mu = self.context2mu(context)
        logvar = self.context2mu(logvar)

    def forward(self, x1, x2, x1_sos, x2_sos):
        c_mask = x1 == PAD  # BL
        r_mask = x2 == PAD

        xx1, xx2 = self.emb(x1), self.emb(x2)

        xx1, _ = self.encoder_rnn(xx1)
        xx2, _ = self.encoder_rnn(xx2)

        z1 = self.context2latent(xx1[:, -1, :])
        z2 = self.context2latent(xx1[:, -1, :])

        xx1_sos = self.emb(x1_sos)
        xx2_sos = self.emb(x2_sos)

        xx1_hat, _ = self.decoder_rnn(xx1_sos, (z1.unsqueeze(0), torch.zeros_like(z1.unsqueeze(0))))
        xx2_hat, _ = self.decoder_rnn(xx2_sos, (z2.unsqueeze(0), torch.zeros_like(z2.unsqueeze(0))))

        logits1 = self.fc2vocab(xx1_hat)
        logits2 = self.fc2vocab(xx2_hat)

        m = torch.cat([z1, z2, z1 - z2, z1 * z2], -1)
        y_hat = self.match(m).squeeze(-1)

        return y_hat, logits1, logits2
# %%

# model = ESIM_VAE()
# model = ESIM()
model = LSTM_AE()

model = model.to(device)

LR = 10e-4


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
    f1 = ((2 * p * r) / (p + r).clip(EPSILON)).mean()
    accu = m.diagonal().sum() / m.sum().clip(EPSILON)
    return p, r, f1, accu


optimizer = torch.optim.Adam(lr=LR, params=model.parameters())

# train_weights = (1 / train_dataset.labels.value_counts(normalize=True))
# train_weights = train_weights.to_numpy() / train_weights.sum()
# train_weights = torch.Tensor(train_weights).to(device)

# progress_bar = tqdm(range(1, N_EPOCH + 1))
# for i_epoch in progress_bar:
for i_epoch in range(1, N_EPOCH + 1):
    model.train()
    loss_total = 0
    match_nll_total = 0
    rec_nll_total = 0
    total = 0
    total_supervised = EPSILON
    train_metrics_supervised = np.zeros((NUM_CLASS, NUM_CLASS))

    alpha = 0

    progress_bar = tqdm(train_data_loader)
    for batch in progress_bar:
        # for x1, x2, y in train_data_loader:
        optimizer.zero_grad()

        batch = tuple(i.to(device) for i in batch)
        x1, x2, x1_sos, x2_sos, x1_eos, x2_eos, y = batch

        y_hat, logits1, logits2 = model(x1, x2, x1_sos, x2_sos)

        batch_size = y.shape[0]
        batch_n_supervised = torch.sum(y != -1).item()

        if batch_n_supervised > 0:
            if NUM_CLASS == 2:
                match_nll = F.binary_cross_entropy_with_logits(y_hat[y != -1], y[y != -1].float(), reduction='sum')
            else:
                match_nll = F.cross_entropy(y_hat[y != -1], y[y != -1], reduction='sum')
        else:
            match_nll = torch.zeros(1).to(device)

        # rec_nll_1 = torch.zeros(1).to(device)
        rec_nll_1 = F.cross_entropy(logits1.view(-1, VOCAB_LEN), x1_eos.view(-1), reduction='sum', ignore_index=PAD)
        # rec_nll_2 = torch.zeros(1).to(device)
        rec_nll_2 = F.cross_entropy(logits2.view(-1, VOCAB_LEN), x2_eos.view(-1), reduction='sum', ignore_index=PAD)

        alpha = min(1, alpha + 0.1)
        loss = alpha * match_nll + .05 * rec_nll_1 + .05 * rec_nll_2


        loss.backward()
        optimizer.step()

        rec_nll_total += (rec_nll_1.item() + rec_nll_2.item()) / 2
        loss_total += loss.item()
        total += batch_size

        if batch_n_supervised > 0:
            total_supervised += batch_n_supervised
            train_metrics_supervised += calc_confusion_matrix(y, y_hat, NUM_CLASS)
            match_nll_total += match_nll.item()

        metrics = (
            loss_total / total,
            match_nll_total / total_supervised,
            rec_nll_total / total,
            *calc_metrics(train_metrics_supervised),
        )

        progress_bar.set_description(Fore.RESET + '[ EP {0:02d} ]'
                                     # '[ TRN LS: {1:.3f} PR: {2:.3f} F1: {4:.3f} AC: {5:.3f}]'
                                     '[ TRN LS: {1:.3f} MA: {2:.3f} RC: {3:.3f} AC: {7:.3f} ]'
                                     .format(i_epoch, *metrics))

    model.eval()

    loss_total_test = 0
    match_nll_total_test = 0
    rec_nll_total_test = 0
    total_test = 0
    test_metrics = np.zeros((NUM_CLASS, NUM_CLASS))

    with torch.no_grad():
        progress_bar = tqdm(dev_data_loader)
        for batch in progress_bar:
            # for x1, x2, y in dev_data_loader:

            batch = tuple(i.to(device) for i in batch)

            x1, x2, x1_sos, x2_sos, x1_eos, x2_eos, y = batch

            y_hat, logits1, logits2 = model(x1, x2, x1_sos, x2_sos)

            if NUM_CLASS == 2:
                match_nll = F.binary_cross_entropy_with_logits(y_hat[y != -1], y[y != -1].float(), reduction='sum')
            else:
                match_nll = F.cross_entropy(y_hat[y != -1], y[y != -1], reduction='sum')

            rec_nll_1 = torch.zeros(1).to(
                device)  # F.cross_entropy(logits1.view(-1, VOCAB_LEN), x1_eos.view(-1), reduction='sum')
            rec_nll_2 = torch.zeros(1).to(
                device)  # F.cross_entropy(logits2.view(-1, VOCAB_LEN), x2_eos.view(-1), reduction='sum')

            loss = match_nll + .0 * rec_nll_1 + .0 * rec_nll_2

            loss_total_test += loss.item()
            match_nll_total_test += match_nll.item()
            rec_nll_total_test += (rec_nll_1.item() + rec_nll_2.item()) / 2
            total_test += y.shape[0]
            test_metrics += calc_confusion_matrix(y, y_hat, NUM_CLASS)

            metrics = (
                loss_total_test / total_test,
                match_nll_total_test / total_test,
                rec_nll_total_test / total_test,
                *calc_metrics(test_metrics)
            )

            progress_bar.set_description(Fore.YELLOW + '[ EP {0:02d} ]'
                                         # '[ TST LS: {1:.3f} PR: {2:.3f} F1: {4:.3f} AC: {5:.3f} ]'
                                         '[ DEV LS: {1:.3f} MA: {2:.3f} RC: {3:.3f} AC: {7:.3f} ]'
                                         .format(i_epoch, *metrics))


    model.eval()

    loss_total_test = 0
    match_nll_total_test = 0
    rec_nll_total_test = 0
    total_test = 0
    test_metrics = np.zeros((NUM_CLASS, NUM_CLASS))

    with torch.no_grad():
        progress_bar = tqdm(test_data_loader)
        for batch in progress_bar:
            # for x1, x2, y in dev_data_loader:

            batch = tuple(i.to(device) for i in batch)

            x1, x2, x1_sos, x2_sos, x1_eos, x2_eos, y = batch

            y_hat, logits1, logits2 = model(x1, x2, x1_sos, x2_sos)

            if NUM_CLASS == 2:
                match_nll = F.binary_cross_entropy_with_logits(y_hat[y != -1], y[y != -1].float(), reduction='sum')
            else:
                match_nll = F.cross_entropy(y_hat[y != -1], y[y != -1], reduction='sum')

            rec_nll_1 = torch.zeros(1).to(
                device)  # F.cross_entropy(logits1.view(-1, VOCAB_LEN), x1_eos.view(-1), reduction='sum')
            rec_nll_2 = torch.zeros(1).to(
                device)  # F.cross_entropy(logits2.view(-1, VOCAB_LEN), x2_eos.view(-1), reduction='sum')

            loss = match_nll + .0 * rec_nll_1 + .0 * rec_nll_2

            loss_total_test += loss.item()
            match_nll_total_test += match_nll.item()
            rec_nll_total_test += (rec_nll_1.item() + rec_nll_2.item()) / 2
            total_test += y.shape[0]
            test_metrics += calc_confusion_matrix(y, y_hat, NUM_CLASS)

            metrics = (
                loss_total_test / total_test,
                match_nll_total_test / total_test,
                rec_nll_total_test / total_test,
                *calc_metrics(test_metrics)
            )

            progress_bar.set_description(Fore.YELLOW + '[ EP {0:02d} ]'
                                         # '[ TST LS: {1:.3f} PR: {2:.3f} F1: {4:.3f} AC: {5:.3f} ]'
                                         '[ TST LS: {1:.3f} MA: {2:.3f} RC: {3:.3f} AC: {7:.3f} ]'
                                         .format(i_epoch, *metrics))
    # metrics = (
    #     loss_total / total,
    #     *p_r_(train_metrics),
    #     loss_total_test / total_test,
    #     *p_r_(test_metrics)
    # )
    #
    # progress_bar.set_description('[ EP {0:02d} ]'
    #                              # '[ TRN LS: {1:.3f} PR: {2:.3f} F1: {3:.3f} AC: {4:.3f}]'
    #                              '[ TRN LS: {1:.3f} AC: {5:.3f} ]'
    #                              # '[ TST LS: {5:.3f} PR: {6:.3f} F1: {7:.3f} AC: {8:.3f} ]'
    #                              '[ TST LS: {6:.3f} AC: {10:.3f} ]'
    #                              .format(i_epoch, *metrics))
