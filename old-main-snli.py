# %%
import math
import numpy as np
import gensim

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SNLI
from tqdm import tqdm

BATCH_SIZE = 128
PAD = 1
EMBEDDING_DIM = 100
MAX_LEN = 40
INF = 1e13
NUM_CLASSES = 3
N_EPOCHS = 25

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
inputs = Field(lower=True, tokenize='spacy', fix_length=MAX_LEN, pad_first=True, truncate_first=True, batch_first=True, tokenizer_language='en_core_web_sm')
answers = Field(sequential=False, batch_first=True)

train, dev, test = SNLI.splits(inputs, answers)

inputs.build_vocab(train, dev, test, min_freq=5)
answers.build_vocab(train)

# %%
train_iter, dev_iter, test_iter = BucketIterator.splits(
    (train, dev, test), batch_size=BATCH_SIZE, device=device)


class IterWrap:

    def __init__(self, iterator) -> None:
        super().__init__()
        self.iterator = iterator

    def __iter__(self):
        for batch in self.iterator:
            yield batch.premise, batch.hypothesis, batch.label

    def __len__(self):
        return len(self.iterator)


train_data_loader = IterWrap(train_iter)
dev_data_loader = IterWrap(dev_iter)
test_data_loader = IterWrap(test_iter)

# %%
w2v = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/amir/IIS/Datasets/embeddings/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt.w2vformat',
    binary=True)

embedding_weights = torch.zeros(len(inputs.vocab), EMBEDDING_DIM)
nn.init.normal_(embedding_weights)

unmatch = []
for i, word in enumerate(inputs.vocab.itos):
    if word in w2v and i != PAD:
        embedding_weights[i] = torch.Tensor(w2v[word])
    else:
        unmatch.append(word)

print(100 - len(unmatch) * 100 / len(inputs.vocab.itos), '% embedding match')

embedding_weights.to(device)


def get_emb():
    return nn.Embedding(len(inputs.vocab), EMBEDDING_DIM, padding_idx=PAD, _weight=embedding_weights.clone())


# %%

class DiSAN(nn.Module):

    def __init__(self, dim=EMBEDDING_DIM):
        super().__init__()
        self.emb = get_emb()
        self.Wh = nn.Linear(dim, dim)
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))
        nn.init.uniform_(self.b, -1 / math.sqrt(dim), 1 / math.sqrt(dim))
        self.c = nn.Parameter(torch.Tensor([5.0]))
        self.fw_mask = torch.ByteTensor(np.tri(MAX_LEN, MAX_LEN, dtype='uint8')) \
            .unsqueeze(-1).expand(MAX_LEN, MAX_LEN, dim).to(device)
        self.bw_mask = torch.ByteTensor(np.tri(MAX_LEN, MAX_LEN, dtype='uint8')).t() \
            .unsqueeze(-1).expand(MAX_LEN, MAX_LEN, dim).to(device)
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
        u_fw = self.multi_dim_masked_attention(h, att, self.fw_mask)  # BLE
        u_bw = self.multi_dim_masked_attention(h, att, self.bw_mask)  # BLE
        u = torch.cat([u_fw, u_bw], -1)  # BL(2E)

        att_s = self.Ws(F.elu(self.Ws1(u)))  # BL(2E)
        s_s = (u * att_s).sum(-2)  # B(2E)
        return s_s


class SWEm(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()

    def forward(self, x, _):
        x = self.emb(x)
        return x.mean(1)


class LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.lstm = nn.LSTM(EMBEDDING_DIM, EMBEDDING_DIM, batch_first=True)

    def forward(self, x, _):
        x = self.emb(x)
        x = self.lstm(x)[0][:, -1, :]
        return x


class SNLI(nn.Module):
    def __init__(self, encoder, encoder_dim):
        super().__init__()
        self.encoder = encoder()
        self.fc_out = nn.Sequential(
            nn.Linear(encoder_dim * 4, encoder_dim * 2),
            nn.ELU(),
            nn.Linear(encoder_dim * 2, NUM_CLASSES))

    def forward(self, pre, hyp):
        p_mask = pre == PAD
        h_mask = hyp == PAD
        pre = self.encoder(pre, p_mask)  # BLH
        hyp = self.encoder(hyp, h_mask)  # BLH
        s = torch.cat([pre, hyp, pre - hyp, pre * hyp], -1)
        return self.fc_out(s)


# %%
models = [
    SNLI(DiSAN, 2 * EMBEDDING_DIM).to(device)
    # SNLI(SWEm, EMBEDDING_DIM).to(device)
    # SNLI(LSTM, EMBEDDING_DIM).to(device)
]
criterion = nn.CrossEntropyLoss(reduction='sum')
metrics_history_all = []
for model in models:
    optimizer = torch.optim.Adam(model.parameters())
    metrics_history = []
    progress_bar = tqdm(range(1, N_EPOCHS + 1))
    for i_epoch in progress_bar:
        model.train()
        loss_total = 0
        accu_total = 0
        total = 0
        # progress_bar = tqdm(train_data_loader)
        for pre, hyp, y in train_data_loader:
            y = y - 1  # labels should start at zero !!
            optimizer.zero_grad()
            prediction = model(pre, hyp)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = (torch.argmax(prediction, 1).long() == y.long()).sum().item()

            batch_size = y.size(0)
            loss_total += loss.item()
            accu_total += acc
            total += batch_size

        with torch.no_grad():
            model.eval()
            loss_total_test = 0
            accu_total_test = 0
            total_test = 0
            for pre, hyp, y in test_data_loader:
                y = y - 1  # labels should start at zero !!
                prediction = model(pre, hyp)
                loss = criterion(prediction, y.long())

                with torch.no_grad():
                    acc = (torch.argmax(prediction, 1).long() == y.long()).sum().item()

                batch_size = y.size(0)
                loss_total_test += loss.item()
                accu_total_test += acc
                total_test += batch_size

        metrics = (
            loss_total / (total * NUM_CLASSES),
            accu_total / total,
            loss_total_test / (total_test * NUM_CLASSES),
            accu_total_test / total_test
        )
        progress_bar.set_description(
            "[ TRAIN LSS: {:.3f} ACC: {:.3f} ][ TEST LSS: {:.3f} ACC: {:.3f} ]".format(*metrics)
        )
        metrics_history.append(metrics)

    metrics_history_all.append(metrics_history)
