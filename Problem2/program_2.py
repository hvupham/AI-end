# -*- coding: utf-8 -*-
"""P2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ekYuRmY45V18XGLAWwzhVI7g_YFmbjSs

# - RNN + LSTM + Glob + dropout
"""

#!pip install torchtext = 0.4.0

"""## import lib"""

import torch
from spacy.tokenizer import Tokenizer
# from torchtext import data
# from torchtext import datasets
from torchtext import data
from torchtext import datasets

# SEED = 11
# torch.manual_seed(SEED)                         ## Reproducibility
torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 20000
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BIDIRECTIONAL = True

EMBEDDING_DIM = 128
NUM_LAYERS = 2
HIDDEN_DIM = 128
OUTPUT_DIM = 1

# TEXT = data.Field(tokenize = 'spacy', include_lengths = True)   ## Text field
# LABEL = data.LabelField(dtype = torch.float)                    ## Label Field

"""## Preparing data"""

#!pip install en-core-web-sm

import random
import torchtext
from torchtext import data
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

TEXT = data.Field(tokenizer,
                  include_lengths=True) # necessary for packed_padded_sequence
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(RANDOM_SEED),
                                          split_ratio=0.8)

# TEXT = data.Field(tokenize='spacy',
#                   include_lengths=True) # necessary for packed_padded_sequence
# LABEL = data.LabelField(dtype=torch.float)

# train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# train_data, valid_data = train_data.split(random_state=random.seed(RANDOM_SEED),
#                                           split_ratio=0.8)

print(f'Num Train: {len(train_data)}')
print(f'Num Valid: {len(valid_data)}')
print(f'Num Test: {len(test_data)}')

TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print(f'Vocabulary size: {len(TEXT.vocab)}')
print(f'Number of classes: {len(LABEL.vocab)}')

train_loader, valid_loader, test_loader = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True, # necessary for packed_padded_sequence
    device=DEVICE)

print('Train')
for batch in train_loader:
    print(f'Text matrix size: {batch.text[0].size()}')
    print(f'Target vector size: {batch.label.size()}')
    break

print('\nValid:')
for batch in valid_loader:
    print(f'Text matrix size: {batch.text[0].size()}')
    print(f'Target vector size: {batch.label.size()}')
    break

print('\nTest:')
for batch in test_loader:
    print(f'Text matrix size: {batch.text[0].size()}')
    print(f'Target vector size: {batch.label.size()}')
    break

"""## Build the model"""

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers = n_layers,
                           bidirectional = bidirectional,
                           dropout = dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedding = self.embedding(text)    ## shape = (sent_length, batch_size)
        embedded = self.dropout(embedding)  ## shape = (sent_length, batch_size, emb_dim)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)    ## pack sequence

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)        ## unpack sequence

        ## output shape = (sent_len, batch_size, hid_dim * num_directions)
        ## output over padding tokens are zero tensors

        ## hidden shape = (num_layers * num_directions, batch_size, hid_dim)
        ## cell shape = (num_layers * num_directions, batch_size, hid_dim)

        ## concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        ## and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) ## shape = (batch_size, hid_dim * num_directions)

        return self.fc(hidden)

VOCABULARY_SIZE = 20000
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BIDIRECTIONAL = True

EMBEDDING_DIM = 100  # 128
NUM_LAYERS = 2
HIDDEN_DIM = 128 # 256
OUTPUT_DIM = 1
DROPOUT = 0.4
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

INPUT_DIM = len(TEXT.vocab)

torch.manual_seed(RANDOM_SEED)
model = Model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"There are {train_params} trainable parameters")

"""##  Replace initial embedding with pretrained embedding"""

TEXT.vocab.vectors.size()

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

"""## Replace and with zeros (they were initialized with the normal distribution)"""

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

"""## Train the model"""

import torch
import torch.nn.functional as F
from torchtext import data

from torchtext import datasets
import time
import random

torch.backends.cudnn.deterministic = True

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    accuracy = correct.sum() / len(correct)
    return accuracy

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    prediction = torch.round(torch.sigmoid(prediction))
    correct = (prediction == ground_truth).float() #convert into float for division

    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp = 0      ## true positive
    tn = 0      ## true negative
    fp = 0      ## false positive
    fn = 0      ## false negative

    for i in range(len(prediction)):
        if prediction[i] == True and ground_truth[i] == True:
            tp += 1
        if prediction[i] == True and ground_truth[i] == False:
            fp += 1
        if prediction[i] == False and ground_truth[i] == True:
            fn += 1
        if prediction[i] == False and ground_truth[i] == False:
            tn += 1

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2 * (precision * recall)/(precision + recall)

    return precision, recall, f1, accuracy

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_accuracy = 0

    model.train()
    for batch in iterator:

        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths.to('cpu')).squeeze(1)

        loss = criterion(predictions, batch.label)
        accuracy = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:

            text, text_lengths = batch.text
            predictions = model(text, text_lengths.to('cpu')).squeeze(1)
            loss = criterion(predictions, batch.label)

            accuracy = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)

def metrics(model, iterator, criterion):

    epoch_loss = 0
    epoch_f1 = 0

    tp = tn = fp = fn = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:

            text, text_lengths = batch.text
            predictions = model(text, text_lengths.to('cpu')).squeeze(1)
            loss = criterion(predictions, batch.label)

            precision, recall, f1, accuracy = binary_classification_metrics(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_f1 += f1

    return epoch_loss / len(iterator), epoch_f1 / len(iterator)

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr = 0.001)

criterion = nn.BCEWithLogitsLoss()      ## use GPU
criterion = criterion.to(DEVICE)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):

    start_time = time.time()

    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model, 'model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_accuracy*100:.2f}%')

test_loss, test_acc = evaluate(model, test_loader, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')