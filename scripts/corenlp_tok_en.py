#!/usr/bin/python
"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

import re
import sys
import html
import multiprocessing
#from nltk.parse.corenlp import CoreNLPParser
from stanfordcorenlp import StanfordCoreNLP


parsers = {'en': StanfordCoreNLP(r'~/distance_transformer/data-bin/stanford-corenlp-full-2018-10-05',lang='en'),
           'de': StanfordCoreNLP(r'~/distance_transformer/data-bin/stanford-corenlp-full-2018-10-05',lang='de')}


def tokenize(sent, lang):
    sent = html.unescape(sent.strip())
    sent = re.sub('([\.!?])([A-Z])', '\g<1> \g<2>', sent)
    sent = ' '.join(parsers[lang].word_tokenize(sent))
    return sent


def tokenize_en(sent):
    return tokenize(sent, 'en')

if __name__ == "__main__":
    path = sys.argv[1]

    pool = multiprocessing.Pool()

    lang = 'en'
    with open(path + '.' + lang) as f:
        en_lines = f.readlines()

    en_tok = pool.map(tokenize_en, en_lines)

    pool.close()

    with open(path + '.tok.en', 'w') as f:
        for line in en_tok:
            f.write(line + '\n')

