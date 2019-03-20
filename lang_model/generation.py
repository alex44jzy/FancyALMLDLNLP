import torch
from train import generate, get_dataset
import config
from torch.autograd import Variable
import os


def generate(input_word, dataset_p, model_p, word_len=100, temperature=1.0):
    model = torch.load(model_p)
    dataset = get_dataset(dataset_p)
    model.eval()
    hidden = (Variable(torch.zeros(config.num_layers, 1, config.hidden_size)).to(device),
              Variable(torch.zeros(config.num_layers, 1, config.hidden_size)).to(device))  # batch_size为1
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


if __name__ == '__main__':
    input_word = 'player'
    data_path = './data/hamlet.txt'
    model_folder_path = "./model_save/"
    files = os.listdir(model_folder_path)
    if not len(files):
        print("There is no model in the model_save folder, please train the model first!")
    else:
        model_file = files[-1]
        model_path = model_folder_path + model_file
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        word_lt = generate(input_word, data_path, model_path)
        print(' '.join(word_lt))
