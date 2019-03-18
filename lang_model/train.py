from data_loader import LangDataset
from data_process import process
from model import RNNLM
import torch
from tqdm import tqdm
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import math


def get_dataset_dataloader(path, batch_size):
    print("Start load data .................")
    # read in data
    input_sents, output_sents = process(path)
    print("Load data finished ..............")

    lang_dataset = LangDataset(input_sents, output_sents)
    dataloader = DataLoader(dataset=lang_dataset, batch_size=batch_size, shuffle=False)
    return lang_dataset, dataloader, input_sents, output_sents


def normalize_sizes(y_pred, y_true):
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


# TODO: add the mask to be ignored
def sequence_loss(y_pred, y_true):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


def generate(model, dataset, input_word, word_len=100, temperature=1.0):
    model.eval()
    hidden = (Variable(torch.zeros(num_layers, 1, hidden_size)).to(device),
              Variable(torch.zeros(num_layers, 1, hidden_size)).to(device))  # batch_size为1
    start_idx = dataset.vectorize(dataset.vocab, [input_word])
    input_tensor = torch.stack([start_idx] * 1)
    input = input_tensor.to(device)
    word_list = [input_word]
    for i in range(word_len):  # generate word by word

        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()
        # get the 1st biggest prob index
        word_idx = torch.multinomial(word_weights, 1)[0]
        if word_idx == 2:
            break
        input.data.fill_(word_idx)  # put new word into input
        word = dataset.unvectorize(dataset.vocab, [word_idx.item()])

        word_list.append(word[0])
    return word_list


def train():
    lang_dataset, dataloader, input_sents, output_sents = \
        get_dataset_dataloader(path, batch_size)
    vocab_size = len(lang_dataset.vocab)

    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    train_acc = []

    for epoch in range(num_epochs):
        states = (Variable(torch.zeros(num_layers, batch_size, hidden_size)).to(device),
                  Variable(torch.zeros(num_layers, batch_size, hidden_size)).to(device))

        running_loss = 0.0
        running_acc = 0.0
        model.train()
        batch_index = 0
        for data_dict in tqdm(dataloader):
            batch_index += 1
            optimizer.zero_grad()
            x = data_dict['x'].to(device)
            y = data_dict['y'].to(device)
            y_pred, states = model(x, states)
            loss = sequence_loss(y_pred, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += (loss.item() - running_loss) / batch_index
            acc_t = compute_accuracy(y_pred, y, mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        print('Epoch = %d, Train loss = %f, Train accuracy = %f, Train perplexity = %f' % (
            epoch, running_loss, running_acc, math.exp(running_loss)))
        train_loss.append(running_loss)
        train_acc.append(running_acc)
        print(' '.join(generate(model, lang_dataset, 'the')))

    return train_loss, train_acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = './data/wiki.train.tokens'
    batch_size = 40
    embed_size = 128
    hidden_size = 512
    num_layers = 2
    dropout_p = 0.1
    num_epochs = 5
    learning_rate = 0.003
    mask_index = 0
    loss_list, acc_list = train()
