import torch.nn as nn
from torch.nn import functional as F


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout_p=0.5):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self._dropout_p = dropout_p

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        # Forward propagate LSTM
        out, h = self.lstm(x, h)
        batch_size, seq_size, hidden_size = out.shape

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(batch_size * seq_size, hidden_size)

        # apply dropout
        out = self.fc(F.dropout(out, p=self._dropout_p))
        out_feat = out.shape[-1]
        out = out.view(batch_size, seq_size, out_feat)
        return out, h
