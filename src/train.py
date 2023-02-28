"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import warnings
import time
from functools import partial

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import register_data_args

from model import *
from utils import *
from data import load_data, gen_index
from configs import *


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):
    data = load_data(args.dataset, DATA_PATH, args.complement, True)

    g: dgl.DGLGraph = data[0]
    cg = data.complementary_graph
    if args.model == 'mf':
        cg = static_mf_reg(cg, load_path='{}/{}/{}_mf_128.npy'.format(DATA_PATH, args.dataset, args.dataset))
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.to(args.gpu)
        try:
            cg = cg.to(args.gpu)
        except:
            pass

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = g.num_edges()

    # create model
    activation_fn = load_activation_fn(args.activation, args.negative_slope)
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = load_model(args.model,
                       args.num_layers,
                       num_feats,
                       args.num_hidden,
                       n_classes,
                       heads,
                       activation_fn,
                       args.in_drop,
                       args.attn_drop,
                       args.negative_slope,
                       args.residual)
    print(model)
    if args.early_stop:
        # stopper = EarlyStopping(patience=100)
        stopper = EarlyStopper(patience=args.es_patience, in_memory=True)
    if cuda:
        model.to(args.gpu)
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    learning_rate = args.lr
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        # forward
        if args.model != 'gat':
            logits, loss0 = model(g, features, cg)
            loss = loss_fcn(logits[train_mask], labels[train_mask]) + loss0 * args.beta
        else:
            logits = model(g, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        with torch.no_grad():
            if args.fastmode:
                # In fastmode, the drop-out is on.
                val_acc = accuracy(logits[val_mask], labels[val_mask])
            else:
                val_acc = evaluate(model, g, features, labels, val_mask)
                if args.early_stop:
                    if stopper.step(val_acc, model):
                        break
            val_loss = loss_fcn(logits[val_mask], labels[val_mask])
            if args.lr_decay:
                scheduler.step(val_loss)

        if args.model != 'gat':
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " ValLoss {:.4f} | ValAcc {:.4f} | Reg {:.4f} | ETputs(KTEPS) {:.2f}".
                  format(epoch, np.mean(dur), (loss - loss0 * args.beta).item(), train_acc,
                         val_loss.item(), val_acc, loss0.item(), n_edges / np.mean(dur) / 1000))
        else:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " ValLoss {:.4f} | ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                  format(epoch, np.mean(dur), loss.item(), train_acc,
                         val_loss.item(), val_acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(stopper.load_checkpoint())
    val_acc = evaluate(model, g, features, labels, val_mask)
    print("Best Val Accuracy {:.4f}".format(val_acc))
    acc = evaluate(model, g, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
    log_end()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--script", type=str, default='train',
                        choices=['train', 'gen_index', 'save_mf'])
    parser.add_argument("--seed", type=int, default=42,
                        help='A negative value indicates no fixed seed.')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--activation', type=str, default='elu',
                        choices=['leaky_relu', 'elu', 'selu', 'celu', 'sin'],
                        help="activation function")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu for attention computation")
    parser.add_argument('--beta', type=float, default=0.1,
                        help="the weight of the wasserstein loss")
    parser.add_argument('--wass', action='store_true', default=False,
                        help="whether to use the wasserstein loss")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--es-patience', type=int, default=100,
                        help='patience of early-stopping')
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--lr-decay', action="store_true", default=False,
                        help="use learning rate decay or not")
    parser.add_argument('--step-size', type=int, default=50,
                        help="Step size for learning rate decay")
    parser.add_argument('--model', type=str, default='gat',
                        choices=['gat', 'wass', 'l1', 'l2', 'quad', 'attn_smooth',
                                 'attn_s_smooth', 'gcn', 'mf'],
                        help='which model to use')
    parser.add_argument('--complement', type=str, default=None,
                        choices=[None, 'Laplacian', 'lap-sym-norm', 'lap-rw-norm', 'adjacent', 'plain-adj'])
    args = parser.parse_args()
    # To adapt for previous code
    if args.model == 'wass':
        args.wass = True
    if args.wass:
        warnings.warn('This is an unused model', DeprecationWarning)
    if args.fastmode:
        warnings.warn('`--fastmode` is deprecated', DeprecationWarning)
    if args.model == 'mf':
        args.complement = 'plain-adj'

    log_time()
    print(args)
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.script == 'train':
        main(args)
    elif args.script == 'gen_index':
        gen_index(args.dataset, DATA_PATH)
    else:
        raise NotImplementedError
