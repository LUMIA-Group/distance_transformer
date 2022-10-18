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
from tensorboardX import SummaryWriter

import numpy as np
import torch
import time

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from omegaconf import DictConfig
from fairseq.trainer import Trainer
from helpers.nltk_tree import build_nltktree_only_bracket
from nltk import Tree
from stanfordcorenlp import StanfordCoreNLP

import re
from tqdm import tqdm
import matplotlib.pylab as plt

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

def weights_init(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm')!= -1:
        mod.weight.data.normal_(1.0,0.02)
        mod.bias.data.fill_(0)


def create_sentence(src, src_dict, remove_bpe=True):
    res = ""
    if isinstance(src, torch.Tensor):
        if src.is_cuda:
            src = src.detach().cpu()
    src_len = src.shape[0]
    for i in range(src_len):
        tok = src_dict[src[i]]
        if src[i] == 18:
            tok = "''"
        # if tok == '.':
        #     tok = ','
        if tok == '–': # <- STARNGE BUG -->
            tok = '--'
        if tok == '>>':
            tok = "--"
        # if ascii(tok)=="'\\u2013'": # for tok = "-"
        #     tok = "--"

        if not remove_bpe:
            if i == src_len - 1:
                res += tok
            else:
                res += tok + " "
        else:
            if tok.endswith("@@"):
                res += tok[:-2]
            else:
                if i == src_len - 1:
                    res += tok
                else:
                    res += tok + " "

    # remove start / end symbol > still have other symbols ?
    remove_start = 0
    remove_end = 0
    if res.startswith("<s>"):
        res = res[3:]
        remove_start = 1
    if res.endswith("</s>"):
        res = res[:-5]
        remove_end = 1
    return res, remove_start, remove_end


def remove_bpe(sentense, distance):
    if not isinstance(sentense, list):
        print("except type of arg sentense to be len !")
        return
    sent_len = len(sentense)
    dist_wo_bpe = []
    sent_wo_bpe = ""
    for i in range(sent_len):
        token = sentense[i]
        if not token.endswith("@@"):
            sent_wo_bpe += token
            if i != sent_len - 1:
                sent_wo_bpe += " "
            if i < sent_len - 1:
                dist = distance[i]
                dist_wo_bpe.append(dist)
        else:
            sent_wo_bpe += token[:-2]
    return sent_wo_bpe, dist_wo_bpe


def process_str_tree(str_tree):
    pat = re.compile('<[^>]+>')
    res = pat.sub("", str_tree)
    return re.sub('[ |\n]+', ' ', res)



def tree2sameNodes(tree, node_name):
    """
    modify a tree to share same node name, leaves remain unchanged
    :param tree: nltk.Tree
    :param node_name: modified nodes name
    :return: a modified tree
    """
    if isinstance(tree, Tree):
        tree.set_label(node_name)
        for child in tree:
            tree2sameNodes(child, node_name)
    else:
        return


def evalb_test(evalb_path, pred_pth, gt_pth, out_pth, e=10000):
    """
    :param evalb_path : exe path
    :param pred_pth:  prediction txt file
    :param gt_pth: ground_truth txt file
    :param out_pth: output file of EVLB
    :param e: max tolerated error
    :return: F1-score
    """
    param_pth = os.path.join(evalb_path, "sample/sample.prm")
    evalb_exe_pth = os.path.join(evalb_path, "evalb")
    cmd = "{} -p {} -e {} {} {} > {}".format(
        evalb_exe_pth,
        param_pth,
        e,
        pred_pth,
        gt_pth,
        out_pth
    )

    os.system(cmd)


    f = open(out_pth)
    for line in f:
        match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
        if match:
            recall = float(match.group(1))
        match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
        if match:
            precision = float(match.group(1))
        match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
        if match:
            fscore = float(match.group(1))
            break

    res = {
        "recall" : recall,
        "precision" : precision,
        "fscore" : fscore
    }

    return res


def main(cfg: DictConfig) -> None:

    # ---------------- some hyper params ----------------#
    device = 2
    epc_list = range(5, 50, 5)
    subset_name = 'valid'
    checkpoint_pth = '/home/htxue/data/distance-transformer/experimental/checkpoint/DT_unsp_tau_1'
    save_pth = 'experimental/results/F1_compare_new/lm_0.5_new/'
    root = '/home/htxue/data/distance-transformer'
    checkpoint_pth = os.path.join(root, checkpoint_pth)
    save_pth = os.path.join(root, save_pth)
    path = "/home/htxue/data/Distance-Transformer/distance-transformer/src_data/stanford-corenlp-full-2018-10-05"
    gt_done = 0
    invalid_sent_id = [2023]
    k = 10  # part of the dataset, size = dataset_size // k
    evalb_pth = "EVALB/"
    mode = "min"
    # ---------------------------------------------------#

    # nlp = StanfordCoreNLP(path, lang='en')

    if os.path.exists(os.path.join(save_pth, "tree_gt.txt")):
        gt_done = 1

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

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.

    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    model = task.build_model(cfg.model)
    model.apply(weights_init)



    dataset = task.dataset(subset_name)
    world_dict = dataset.src_dict


    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)

    # gt_done = False
    if not gt_done:
        f_gt = open(os.path.join(save_pth, 'tree_gt.txt'), "w")
        dataset_size = dataset.__len__()
        print(dataset_size)
        for sent_id in tqdm(range(dataset_size//k)):

            if sent_id in invalid_sent_id:
                continue

            item = dataset[sent_id]
            src_tokens = item['source']
            src_tokens = src_tokens.unsqueeze(0)

            # world_dict[18] = "\""  # &quot -> "

            sent, rs, re = create_sentence(src=src_tokens[0], src_dict=world_dict, remove_bpe=True)
            tree = Tree.fromstring(nlp.parse(sent))
            # print(tree)
            tree.chomsky_normal_form(childChar="+")
            tree2sameNodes(tree, "Node")
            tree_str = process_str_tree(str(tree))

            f_gt.write(tree_str + "\n")
        f_gt.close()

    F_score_y = []
    F_score_x = []

    for epc in epc_list:
        dataset_size = dataset.__len__()
        model_path = os.path.join(checkpoint_pth, 'checkpoint{}.pt'.format(epc))
        tmp = torch.load(model_path)['model']
        model.load_state_dict(tmp)
        model.eval()
        print("[load] ---epc{} checkpoint---".format(epc))

        f_pred = open(os.path.join(save_pth, "{}_pred.txt".format(epc)), "w")

        for sent_id in tqdm(range(dataset_size//k)):
            if sent_id in invalid_sent_id:
                continue
            time_st = time.time()

            item = dataset[sent_id]
            src_tokens = item['source']
            src_tokens = src_tokens.unsqueeze(0)

            sent_bpe, rs, re = create_sentence(src=src_tokens[0], src_dict=world_dict, remove_bpe=False)
            sent_bpe_list = sent_bpe.split(" ")

            # get distancez
            x, _ = model.encoder.forward_embedding(src_tokens, None)
            x = x.transpose(0, 1)
            x_conv = x.permute(1, 2, 0)
            dist = model.encoder.conv_layers[0](x_conv)  # [1 * 2 * src_len]
            dist = dist.permute(2, 0, 1)  # n-1 * B * 2
            dist = torch.cat((dist[:-4, :, [0]], dist[4:, :, [1]]), dim=-1)  # 0: pre 1: pos N-1 * B * 2
            # sum of pre and post distance

            # dist = dist.sum(dim=1) / 2
            if mode == "pre":
                dist = dist[:, :, 0]
            elif mode == "pos":
                dist = dist[:, :, 1]
            elif mode == "mean":
                dist = dist.mean(dim=-1)
            elif mode == "rand":
                dist = torch.rand(dist[:, :, 1].shape)
            elif mode == 'max':
                dist = dist.max(-1)[0]
            elif mode == 'min':
                dist = dist.min(-1)[0]



            dist = dist.squeeze(-1)
            dist_list = [i.item() for i in dist]

            if rs:
                dist_list = dist_list[1:]
            if re:
                dist_list = dist_list[:-1]

            sent, dist = remove_bpe(sent_bpe_list, distance=dist_list)
            sent_list = sent.split(" ")

            if '.' in sent_list:
                position = sent_list.index(".") + 1
                sent_list = sent_list[:position]
                dist = dist[:(position-1)]

            tree_pred = build_nltktree_only_bracket(dist, sent_list)

            tree2sameNodes(tree_pred, "Node")

            tree_pred_str = process_str_tree(str(tree_pred))

            f_pred.write("(Node " + tree_pred_str + ")")
            f_pred.write("\n")

        f_pred.close()

        res = evalb_test(os.path.join(root, evalb_pth),
                         os.path.join(save_pth, "{}_pred.txt".format(epc)),
                         os.path.join(save_pth, "tree_gt.txt"),
                         os.path.join(save_pth, "tmp_{}.txt".format(epc)),
                         e=10000)
        print("In epc {}".format(epc), "The test score is as follows:")
        print(res)
        F_score_y.append(res['fscore'])
        F_score_x.append(epc)
        plt.plot(F_score_x, F_score_y, "ob:")
        plt.savefig(save_pth + "curve_{}.png".format(mode))
        plt.close()




def cal_F1_main(
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
    cal_F1_main()
