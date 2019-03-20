import torch
from train import generate, get_dataset


def real_time_generator(word, d_path, m_path):
    pass

if __name__ == '__main__':
    input_word = 'player'
    model_path = './model_save/best_model_epoch4_loss_5.112230.pth'
    data_path = './data/hamlet.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    real_time_generator(input_word, data_path, model_path)
