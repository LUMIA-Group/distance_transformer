#!/usr/bin/python
"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

import re
import sys
import html
import numpy as np
from collections import defaultdict
#from nltk.parse.corenlp import CoreNLPParser
from stanfordcorenlp import StanfordCoreNLP

mapper = {'"': '``'}
parsers = {'en': StanfordCoreNLP(r'~/distance_transformer/data-bin/stanford-corenlp-full-2018-10-05',lang='en'),
           'de': StanfordCoreNLP(r'~/distance_transformer/data-bin/stanford-corenlp-full-2018-10-05',lang='de')}


def tokenize(sent):
    tokens = html.unescape(sent.strip()).split()
    tokens = list(map(lambda t: mapper.get(t, t), tokens))
    return tokens


def remove_bpe(sent, separator='@@'):
    tokens = tokenize(sent)
    word, words = [], []
    for tok in tokens:
        if tok.endswith(separator):
            tok = tok.strip(separator)
            word.append(tok)
        else:
            word.append(tok)
            words.append(''.join(word))
            word = []
    sentence = ' '.join(words)
    return sentence


def annotate(sentences):

    def fix_tok(tok):
        if tok != '...' and tok != '.' and tok[-1] == '.':
            tok = tok[:-1]
        tok = tok.replace("n't", "not")
        tok = tok.replace("&apos;", "'")
        tok = re.sub("(.*)-[A-Za-z]\.[A-Za-z]\.[A-Za-z]", "\g<1>", tok)
        tok = re.sub("(.*)-[A-Za-z]\.[A-Za-z]", "\g<1>", tok)
        tok = re.sub("'[Nn]'", "and", tok)
        tok = re.sub("-+", "-", tok)
        tok = re.sub("^-(.+)", "\g<1>", tok)
        tok = re.sub("'([A-Za-ce-rt-z])", "\g<1>", tok)
        tok = re.sub("'([A-Za-z].+)", "\g<1>", tok)
        return tok

    word_tags = []

    for ix, sent in enumerate(sentences):
        sent_token_tags = []
        splits = list(map(fix_tok, sent.split()))
        parses = list(dep_parser.dependency_parse(splits))
        dep_sent = [line.split('\t')[-1] for line in parses[0].to_conll(4).split('\n') if line != '']
        sent_token_tags.extend(dep_sent)
        dep_len = len(dep_sent)
        while dep_len < len(splits):
            # Multiple sentences detected
            parses = list(dep_parser.dependency_parse(splits[dep_len:]))
            dep_sent = [line.split('\t')[-1] for line in parses[0].to_conll(4).split('\n') if line != '']
            sent_token_tags.extend(dep_sent)
            dep_len += len(dep_sent)

        split1 = [s for s in sentences[ix].strip().split() if s != '']
        word_tags.append([(html.unescape(split1[i]), tag) for i, tag in enumerate(sent_token_tags)])

    return word_tags


def tags_to_bpe(sent, word_tags, separator='@@'):
    tokens = tokenize(sent)
    t_ix = 0
    word = ''
    sent_tags = []
    word2tok = defaultdict(list)
    for wix, (untok_word, tag) in enumerate(word_tags):
        while word != untok_word:
            tok = tokens[t_ix]
            tok = tok[:-len(separator)] if tok.endswith(separator) else tok
            word += tok
            word = html.unescape(word) if word.startswith('&') and word.endswith(';') else word
            sent_tags.append(tag)
            word2tok[wix].append(t_ix)
            t_ix += 1
        word = ''
    return ' '.join(sent_tags)


if __name__ == "__main__":
    lang = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3]
    size = int(sys.argv[4])
    i = int(sys.argv[5])

    dep_parser = parsers[lang]

    # Read BPE-formatted sentences
    with open(infile, 'r', encoding='utf-8') as f:
        bpe_sents = f.readlines()
    bpe_sents = bpe_sents[i*size: min(len(bpe_sents),(i+1)*size)]

    # Attempt reconstructing original data
    recovered_sents = list(map(remove_bpe, bpe_sents))

    # Annotate tokens from reconstructed words
    word_tags = annotate(recovered_sents)

    # Annotate BPE tokens
    bpe_tags = list(map(tags_to_bpe, bpe_sents, word_tags))

    # Write tags to file
    with open(outfile, 'w', encoding='utf-8') as f:
        for tags in bpe_tags:
            f.write(tags + '\n')

    # Test BPE tokens-tags matching
    for l1, l2 in zip(bpe_sents, bpe_tags):
        assert len(l1.split()) == len(l2.split())

