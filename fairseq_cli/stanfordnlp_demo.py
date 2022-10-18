from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
import time
import re

def process_str_tree(str_tree):
    # pat = re.compile('<[^>]+>')
    # res = pat.sub("", str_tree)
    return re.sub('[ |\n]+', ' ', str_tree)


path = "~/distance_transformer/data/stanford-corenlp-full-2018-10-05"
nlp = StanfordCoreNLP(path, lang='zh')

def output_tree(s):
    #print('Constituency Parsing:', nlp.parse(s)) #语法树
    tree = Tree.fromstring(nlp.parse(s))
    tree.pretty_print()
