import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from gensim.corpora.dictionary import Dictionary


class LangDataset(Dataset):
    def __init__(self, src_sents, trg_sents, max_len=-1):
        self.src_sents = src_sents
        self.trg_sents = trg_sents

        # Create the vocabulary for both the source and target.
        self.vocab = Dictionary(src_sents + trg_sents)

        # Patch the vocabularies and add the <pad> and <unk> symbols.
        special_tokens = {'<pad>': 0, '<unk>': 1, '</s>': 2}
        self.vocab.patch_with_special_tokens(special_tokens)

        # Keep track of how many data points.
        self._len = len(src_sents)

        if max_len < 0:
            # If it's not set, find the longest text in the data.
            max_src_len = max(len(sent) for sent in src_sents)
            self.max_len = max_src_len
        else:
            self.max_len = max_len

    def pad_sequence(self, vectorized_sent, max_len):
        # To pad the sentence:
        # Pad left = 0; Pad right = max_len - len of sent.
        pad_dim = (0, max_len - len(vectorized_sent))
        return F.pad(vectorized_sent, pad_dim, 'constant')

    def __getitem__(self, index):
        vectorized_src = self.vectorize(self.vocab, self.src_sents[index])
        vectorized_trg = self.vectorize(self.vocab, self.trg_sents[index])
        return {'x': self.pad_sequence(vectorized_src, self.max_len),
                'y': self.pad_sequence(vectorized_trg, self.max_len),
                'x_len': len(vectorized_src),
                'y_len': len(vectorized_trg)}

    def __len__(self):
        return self._len

    def vectorize(self, vocab, tokens):
        """
        :param tokens: Tokens that should be vectorized.
        :type tokens: list(str)
        """
        # See https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary.doc2idx
        # Lets just cast list of indices into torch tensors directly =)
        return torch.tensor(vocab.doc2idx(tokens, unknown_word_index=1))

    def unvectorize(self, vocab, indices):
        """
        :param indices: Converts the indices back to tokens.
        :type tokens: list(int)
        """
        return [vocab[i] for i in indices]
