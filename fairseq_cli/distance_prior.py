#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

import dill
import numpy as np
import torch
import time
import threading
from multiprocessing import Pool
import multiprocessing
import concurrent.futures

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    tasks,
    utils,
)




from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from omegaconf import DictConfig
from helpers.nltk_tree import build_nltktree_only_bracket
from nltk import Tree
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm


import re
import nltk
import matplotlib.pylab as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

def build_nltktree_only_bracket(depth, sen):
    """stags are the stanford predicted tags present in the train/valid/test files.
    """
    assert len(sen) > 0
    assert len(depth) == len(sen) - 1, ("%s_%s" % (len(depth), len(sen)))

    if len(sen) == 1:

        # if stags, put the real stanford pos TAG for the word and leave the
        # unary chain on top.
        word = str(sen[0])
        word = nltk.Tree('leaf', [word])
        assert isinstance(word, nltk.Tree)
        return word
    else:
        idx = np.argmax(depth)
        node0 = build_nltktree_only_bracket(depth[:idx], sen[:idx + 1])
        node1 = build_nltktree_only_bracket(depth[idx + 1:], sen[idx + 1:])

        if node0.label() != '<empty>' and node1.label() != '<empty>':
            tr = [node0, node1]
        elif node0.label() == '<empty>' and node1.label() != '<empty>':
            tr = [c for c in node0] + [node1]
        elif node0.label() != '<empty>' and node1.label() == '<empty>':
            tr = [node0] + [c for c in node1]
        elif node0.label() == '<empty>' and node1.label() == '<empty>':
            tr = [c for c in node0] + [c for c in node1]
        tr = nltk.Tree('node', tr)
        return tr

def tree2list(tree, parent_arc=[]):
    if isinstance(tree, nltk.Tree):
        label = tree.label()
        if isinstance(tree[0], nltk.Tree):
            label = re.split('-|=', tree.label())[0]
        root_arc_list = parent_arc + [label]
        root_arc = '+'.join(root_arc_list)
        if len(tree) == 1:
            root, arc, tag = tree2list(tree[0], parent_arc=root_arc_list)
        elif len(tree) == 2:
            c0, arc0, tag0 = tree2list(tree[0])
            c1, arc1, tag1 = tree2list(tree[1])
            root = [c0, c1]
            arc = arc0 + [root_arc] + arc1
            tag = tag0 + tag1
        else:
            c0, arc0, tag0 = tree2list(tree[0])
            c1, arc1, tag1 = tree2list(nltk.Tree('<empty>', tree[1:]))
            if bin == 0:
                root = [c0] + c1
            else:
                root = [c0, c1]
            arc = arc0 + [root_arc] + arc1
            tag = tag0 + tag1
        return root, arc, tag
    else:
        if len(parent_arc) == 1:
            parent_arc.insert(0, '<empty>')
        # parent_arc[-1] = '<POS>'
        del parent_arc[-1]
        return str(tree), [], ['+'.join(parent_arc)]


def cal_distance(root):
    if isinstance(root, list):
        dist_list = []
        depth_list = []
        for child in root:
            dist, depth = cal_distance(child)
            dist_list.append(dist)
            depth_list.append(depth)

        max_depth = max(depth_list)

        out = dist_list[0]
        for dist in dist_list[1:]:
            out.append(max_depth)
            out.extend(dist)
        return out, max_depth + 1
    else:
        return [], 1


def remove_bpe(word_list):
    """

    :param word_list: ['i', 'am', 'a', 'go@@', 'd', 'boy', '.']
    :return:
    """
    new_list = []
    bpe_info = []
    tok = ""
    bpe_count = 0
    for item in word_list:
        if item.endswith("@@"):
            if tok == "":
                bpe_info.append([len(new_list), 0])
            item_remove_bpe = item[:-2]
            tok += item_remove_bpe
            bpe_count += 1
        else:
            new_list.append(tok+item)
            if len(tok) > 0:
                bpe_count += 1
                bpe_info[-1][1] = bpe_count
            bpe_count = 0
            tok = ""
    return new_list, bpe_info


def remove_space_parse(nlp,word_list):
    """

    :param word_list: ['i', 'am', 'a', 'go@@', 'd', 'boy', '.']
    :return:
    """
    new_list = []
    bpe_info = []
    tok = ""
    bpe_count = 0
    sent_str=''.join(word_list)
    token_list=nlp.word_tokenize(sent_str)
    word_list=[]
    for token in token_list:
        if len(token)>1:
           for i,character in enumerate(token):
            if i<len(token)-1:
              word_list.append(character+"@@")
            else:
              word_list.append(character)
        else:
           word_list.append(token)
    for item in word_list:
        if item.endswith("@@"):
            if tok == "":
                bpe_info.append([len(new_list), 0])
            item_remove_bpe = item[:-2]
            tok += item_remove_bpe
            bpe_count += 1
        else:
            new_list.append(tok+item)
            if len(tok) > 0:
                bpe_count += 1
                bpe_info[-1][1] = bpe_count
                # print(bpe_info)
            bpe_count = 0
            tok = ""
    return new_list, bpe_info


def valid_word(word):
    if "&" in word:
        return False

    return True

def is_cutting_word(word):
    return word in ['.', '!', '?','。','！','？']


def dividing_sent(word):
    res = []
    tmp = []
    d_point = []
    for i, item in enumerate(word):
        if not is_cutting_word(item):
            tmp.append(item)
        else:
            tmp.append(item)
            res.append(tmp)
            tmp = []
            d_point.append(i)
    if len(tmp):
        res.append(tmp)
        d_point.append(len(word))
    return res, d_point

def list2sent(word_list):
    res = ""
    for item in word_list:
        res += (item + " ")
    return res[:-1]

def list_fill_up(raw_key, raw_val, cur_key, cur_val):
    """

    :param raw_key: N
    :param raw_val: N-1
    :param cur_key: nlp core word list N
    :param cur_val: nlp core distance list N - 1
    :return:
    """
    raw_idx = 0
    for i in range(len(cur_key) - 1):
        if cur_key[i] in raw_key:

            while raw_idx < len(raw_val) and cur_key[i] != raw_key[raw_idx]:
                raw_idx += 1

            if raw_idx >= len(raw_val):
                break
            raw_val[raw_idx] = cur_val[i]


def multi_insertion(a, b, v):
    """

    :param a: tgt_list
    :param b: sorted [(ins_pos, ins_num)]
    :param v: ins_val
    :return:
    """
    base = 0
    for item in b:
        pos, num = item[0], item[1]
        pos += base
        base += (num-1)
        a = a[:pos] + [v] * (num - 1) + a[pos:]
    return a


def fill_up_bpe(cur_distance, bpe_info):
    """

    :param cur_distance:
    :param bpe_info: sorted
    :return:
    """
    return multi_insertion(cur_distance, bpe_info, v=0)





def sent_distance(sent):
    """
    input sent list, output distance list
    :param sent:
    :return:
    """
    pass


def thread_function(sent_id):
    item = dataset[sent_id]
    src_tokens = item['source']
    src_tokens = src_tokens

    if isinstance(src_tokens, torch.Tensor):
        if src_tokens.is_cuda:
            src_tokens = src_tokens.detach().cpu()

    num_token = src_tokens.shape[0]
    raw_word_list = [word_dict[src_tokens[i]] for i in range(num_token)]

    word_list, bpe_info = remove_bpe(raw_word_list)  #
    distance_list = []

    if word_list[-1] == "</s>":
        word_list = word_list[:-1]

    # detect ".", "!", "?", "。", "！", "？"which may cut the sent

    if len(word_list) >= 1:
        if word_list[-1] in [',', ';', '.', '--', ':', '-', '...', '..']:
            word_list[-1] = '.'
        elif word_list[-1] in ['。', '；', '，', '——', '：', '。。。', '。。']:
            word_list[-1] = '。'

    sent_list, d_point = dividing_sent(word_list)

    assert len(sent_list) == len(d_point)

    for i, sent in enumerate(sent_list):
        sent_valid = []

        for item in sent:
            if valid_word(item):
                sent_valid.append(item)

        if len(sent_valid) <= 1:
            if i < len(sent_list) - 1:
                distance_list += [-1]
            continue

        sent_str = list2sent(sent_valid)

        try:
            tree = Tree.fromstring(nlp.parse(sent_str))
            # output_tree(sent_str)
            tree.chomsky_normal_form(childChar="+")
            tree_list = tree2list(tree)[0]
            distance = cal_distance(tree_list)[0]
            leaves = tree.leaves()

            assert len(leaves) == (len(distance) + 1)

            distance_list_part = [0 for i in range(len(sent) - 1)]

            # fill up the distance list
            list_fill_up(raw_key=sent,
                            raw_val=distance_list_part,
                            cur_key=leaves,
                            cur_val=distance)
        except:
            distance_list_part = [1 for i in range(len(sent) - 1)]

        distance_list += distance_list_part

        if i < len(sent_list) - 1:
            distance_list += [-1]  # dviding part distance -> +88

    distance_list += [1999]  # for </s>

    distance_list = fill_up_bpe(distance_list, bpe_info)

    if len(distance_list) < len(raw_word_list) - 1:
        distance_list += [0] * (- len(distance_list) + len(raw_word_list) + 1)
    if len(distance_list) > len(raw_word_list) - 1:
        distance_list = distance_list[:len(raw_word_list) - 1]

    assert len(distance_list) == len(raw_word_list) - 1, "Error on " + str(sent_id)

    distance_list = [item + 1 for item in distance_list]
    distance_list = [999 if item == 0 else item for item in distance_list]

    np.save(split_saving + "/{}.npy".format(sent_id), distance_list)



def main(cfg: DictConfig) -> None:
    global nlp
    global dataset
    global word_dict
    global split_saving

    # ---------------- some hyper params ----------------#
    root = "~/distance_transformer"
    corenlp_path = os.path.join(root, "data-bin", "stanford-corenlp-full-2018-10-05")
    save_path = os.path.join(root, "distance_prior",cfg.task.task_name)
    src_tgt = 'src'
    # ----------------  params domains ----------------#
    assert src_tgt in ['src', 'tgt']

    lang_dict = {'ch': 'zh', 'en': 'en', 'de': 'de', 'fr': 'fr'}
    nlp = StanfordCoreNLP(corenlp_path, lang=lang_dict[cfg.task.source_lang])

    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Setup task, e.g., translation, language modeling, etc.

    task = tasks.setup_task(cfg.task)


    for split in ['train', 'valid', 'test']:
        # ---------------------------------------------------#

        # ----------------- make necessary dir ---------- #
        split_saving = os.path.join(save_path, split)
        if not os.path.exists(split_saving):
            os.makedirs(split_saving)

        # Load valid dataset (we load training data below, based on the latest checkpoint)
        task.load_dataset(split, combine=False, epoch=1)

        dataset = task.dataset(split)
        word_dict = dataset.src_dict if src_tgt == 'src' else dataset.tgt_dict

        # poolsize = 20
        # pool = multiprocessing.Pool(processes=poolsize)
        for i, sent_id in enumerate(tqdm(range(dataset.__len__()))):
            #pool.apply_async(thread_function, (sent_id,))
            if not os.path.exists(os.path.join(split_saving,str(i)+".npy")):
                print(sent_id)
            # if i>=200000 and i<240000:
                thread_function(sent_id)
        # pool.close()
        # pool.join()

        print(split, len(dataset))
        print("ending for execution")


def cal_distance_prior_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    print(cfg)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cal_distance_prior_main()
