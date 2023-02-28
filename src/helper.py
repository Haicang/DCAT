"""
Get the MF results of adj matrix before training of rGAT.
"""
import argparse

import numpy as np
import dgl

from data import load_data, gen_index
from model import *
from configs import *
from utils import log_time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, default='mf',
                        choices=['mf', 'index'],
                        help='mf -- generate fix mf; index -- generate index')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dim', type=int, default=128)

    args = parser.parse_args()
    log_time()
    print(args)
    return args


def train_mf(args):
    data = load_data(args.dataset, DATA_PATH, None, process=False)
    g: dgl.DGLGraph = data[0]
    adj = g.adj(scipy_fmt='coo')
    mf = graph_mf(adj, args.dim)
    np.save('{}/{}/{}_mf_{}'.format(DATA_PATH, args.dataset, args.dataset, args.dim), mf)


if __name__ == '__main__':
    args = get_args()
    if args.seed >= 0:
        np.random.seed(args.seed)

    if args.script == 'mf':
        train_mf(args)
    elif args.script == 'index':
        gen_index(args.dataset, DATA_PATH)
    else:
        raise NotImplementedError
