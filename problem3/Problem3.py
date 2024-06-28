import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchtext

from torch import Tensor
from torch import nn
from torch.nn import Transformer
import math

import requests
import tarfile
from tqdm import tqdm
import time
from prettytable import PrettyTable
from torchtext.data.metrics import bleu_score

import time
torch.manual_seed(0)

def iwslt15(train_test):
  url = "http://github.com/stefan-it/nmt-en-vi/raw/master/data/"
  r = requests.get(url+train_test+"-en-vi.tgz")
  filename = train_test
  with open(filename, 'wb') as f:
    f.write(r.content)
    tarfile.open(filename, 'r:gz').extractall("iwslt15")
iwslt15("train")
iwslt15("test-2013")

f = open("iwslt15/train.en")
train_en = [line.split() for line in f]
f.close()
f = open("iwslt15/train.vi")
train_vi = [line.split() for line in f]
f.close()
f = open("iwslt15/tst2013.en")
test_en = [line.split() for line in f]
f.close()
f = open("iwslt15/tst2013.vi")
test_vi = [line.split() for line in f]
f.close()

for i in range(10):
  print(train_en[i])
  print(train_vi[i])
print("# of lien" , len(train_en), len(train_vi), len(test_en), len(test_vi))

EPOCH = 30
BATCHSIZE = 16
LR = 0.0001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_vocab(train_data, min_freq):
  vocab = {}
  for tokenlist in train_data:
    for token in tokenlist:
      if token not in vocab:
        vocab[token] = 0
      vocab[token] += 1 # Đếm các từ data
  vocablist = [('<unk>',0),('<pad>', 0), ('<cls>', 0), ('<eos>', 0)] # Danh sách các tuple đặc trưng trước
  vocabidx = {} # dict của tuple
  for token, freq in vocab.items(): # Từ đó , với số lượng của từ đó trong train data
    if freq >= min_freq: # Số lượng từ mà > 3
      idx = len(vocablist) # đang là 4
      vocablist.append((token, freq)) #Từ đó , với số lượng của từ đó trong train data
      vocabidx[token] = idx  # Từ điển (dict)
  vocabidx['<unk>'] = 0
  vocabidx['<pad>'] = 1
  vocabidx['<cls>'] = 2
  vocabidx['<eos>'] = 3
  return vocablist, vocabidx

vocablist_en, vocabidx_en = make_vocab(train_en, 3)
vocablist_vi, vocabidx_vi = make_vocab(train_vi, 3)

print('vocab size en: ', len(vocablist_en))
print('vocab size vi: ', len(vocablist_vi))

def preprocess(data, vocabidx):
  rr = []
  for tokenlist in data:
    tkl = ['<cls>']
    for token in tokenlist:
      tkl.append(token if token in vocabidx else '<unk>')
    tkl.append('<eos>')
    rr.append((tkl))
  return rr

train_en_prep = preprocess(train_en, vocabidx_en)
train_vi_prep = preprocess(train_vi, vocabidx_vi)
test_en_prep = preprocess(test_en, vocabidx_en)

for i in range(5):
  print(train_en_prep[i])
  print(train_vi_prep[i])
  print(test_en_prep[i])
  
train_data = list(zip(train_en_prep, train_vi_prep))
train_data.sort(key = lambda x: (len(x[0]), len(x[1])))
test_data = list(zip(test_en_prep, test_en, test_vi))

def make_batch(data, batchsize):
  bb = []
  ben = []
  bvi = []
  for en, vi in data:
    ben.append(en)
    bvi.append(vi)
    if len(ben) >= batchsize:
      bb.append((ben, bvi))
      ben = []
      bvi = []
  if len(ben) > 0:
    bb.append((ben, bvi))
  return bb

train_data = make_batch(train_data, BATCHSIZE)

for i in range(5):
  print(train_data[i])
  
def padding_batch(b):
  maxlen = max([len(x) for x in b])
  for tkl in b:
    for i in range(maxlen - len(tkl)):
      tkl.append('<pad>')

def padding(bb):
  for ben, bvi in bb:
    padding_batch(ben)
    padding_batch(bvi)

padding(train_data)
for i in range(3):
  print(train_data[i])
  
train_data = [([[vocabidx_en[token] for token in tokenlist] for tokenlist in ben],
               [[vocabidx_vi[token] for token in tokenlist] for tokenlist in bvi]) for ben,bvi in train_data]
test_data = [([vocabidx_en[token] for token in enprep], en, vi) for enprep, en, vi in test_data]

for i in range(3):
  print(train_data[i])
for i in range(3):
  print(test_data[i])
  
  
class PositionalEncoding(torch.nn.Module):
  def __init__(self, emb_size: int, dropout: float, maxlen: int = 900):
    super(PositionalEncoding, self).__init__()
    den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
    pos = torch.arange(0, maxlen).reshape(maxlen, 1)
    pos_embedding = torch.zeros((maxlen, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    pos_embedding = pos_embedding.unsqueeze(-2)

    self.dropout = torch.nn.Dropout(dropout)
    self.register_buffer('pos_embedding', pos_embedding)

  def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
      

class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
      

class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.01):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = torch.nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = torch.nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

MODELNAME = 'iwslt15-en-vi-transformer.model'
SRC_VOCAB_SIZE = len(vocablist_en)
TGT_VOCAB_SIZE = len(vocablist_vi)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in model.parameters():
  if p.dim() > 1:
    torch.nn.init.xavier_uniform_(p)
model = model.to(DEVICE)
PAD_IDX = 1
loss_fn = torch.nn.CrossEntropyLoss(ignore_index = PAD_IDX)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == vocabidx_en['<pad>']).transpose(0, 1)
    tgt_padding_mask = (tgt == vocabidx_vi['<pad>']).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
  
  
def train():
  optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
  model.train()
  all_time= 0
  for epoch in range(30):
    start= time.time()
    loss = 0
    for en, vi in train_data:
      en = torch.tensor(en, dtype=torch.int64).transpose(0,1).to(DEVICE)
      vi = torch.tensor(vi, dtype=torch.int64).transpose(0,1).to(DEVICE)
      tgt_input = vi[:-1, :]
      optimizer.zero_grad()
      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(en, tgt_input)
      y = model(en, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
      tgt_out = vi[1:, :]
      batchloss = loss_fn(y.reshape(-1, y.shape[-1]), tgt_out.reshape(-1))
      batchloss.backward()
      optimizer.step()
      loss = loss + batchloss.item()
    end= time.time()
    all_time+=(end-start)
    print("epoch", epoch, ": loss", loss, "Epoch time",end-start)
  torch.save(model.state_dict(), MODELNAME)

def evaluate(model, src, max_len, start_symbol):
    num_tokens = src.shape[0]
    src_mask = torch.zeros((num_tokens, num_tokens),device=DEVICE).type(torch.bool)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    pred = []
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        # next_word = out.squeeze().argmax()
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        # print(prob)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == vocabidx_en['<eos>']:
            break
        pred_y = vocablist_vi[next_word][0]
        pred.append(pred_y)
    return pred


def test():
  model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(DEVICE)
  model.load_state_dict(torch.load(MODELNAME))
  model.eval()
  ref = []
  pred = []
  for enprep, en, vi in test_data:
    input = torch.tensor([enprep], dtype=torch.int64).transpose(0,1).to(DEVICE).view(-1, 1)
    p = evaluate(model, input, 50, vocabidx_en['<cls>'])
    ref.append([vi])
    pred.append(p)
  bleu = bleu_score(pred, ref)
  print("total: ", len(test_data))
  print("bleu: ", bleu)
  
train()
test()