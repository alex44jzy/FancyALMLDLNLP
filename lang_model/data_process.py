import os
import io
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


def tokenize(text):
    END = '</s>'
    tokenized_text = [list(map(str.lower, word_tokenize(sent) + [END])) for sent in sent_tokenize(text)]
    return tokenized_text


# TODO just pick the top 200 sentences
def readin(path):
    if os.path.isfile(path):
        with io.open(path, encoding='utf8') as fin:
            text = fin.read()
    text = text.replace('\n', ' ')
    return text


def construct_input_output(sents):
    input_sents = [words[:-1] for words in sents]
    output_sents = [words[1:] for words in sents]
    return input_sents, output_sents


def process(path):
    list_of_lines = readin(path)
    lines_tokens = tokenize(list_of_lines)
    input, output = construct_input_output(lines_tokens)
    print("Total sentences number is %d" % len(input))
    return input, output


if __name__ == '__main__':
    path = './data/hamlet.txt'
    lines = process(path)
