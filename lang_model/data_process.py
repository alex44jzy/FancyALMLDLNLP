import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

try:  # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize

    # Testing whether it works.
    # Sometimes it doesn't work on some machines because of setup issues.
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except:  # Use a naive sentence tokenizer and toktok.
    import re
    from nltk.tokenize import ToktokTokenizer

    # See https://stackoverflow.com/a/25736515/610569
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    # Use the toktok tokenizer that requires no dependencies.
    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize


def tokenize(lines):
    END = '</s>'
    sents = [str.lower(str(line)).split() + [END] for line in lines]
    return sents


# TODO just pick the top 200 sentences
def readin(path):
    list_lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) == 0:  # 过滤空的行
                continue
            words = line.split() + ['</s>']
            # if 20 < len(words) < 500:
            list_lines.append(line)
    return list_lines[:100]


def construct_input_output(sents):
    input_sents = [words[:-1] for words in sents]
    output_sents = [words[1:] for words in sents]
    return input_sents, output_sents


def process(path):
    list_of_lines = readin(path)
    lines_tokens = tokenize(list_of_lines)
    input, output = construct_input_output(lines_tokens)
    return input, output


if __name__ == '__main__':
    path = './data/hamlet.txt'
    lines = process(path)
    print(lines)
