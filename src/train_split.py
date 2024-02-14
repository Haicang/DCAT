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

import dgl
import numpy as np
import torch
from dgl.data import register_data_args
from sklearn.metrics import normalized_mutual_info_score, completeness_score

from configs import *
from model import *
from utils import *
from data import load_data, load_random_splits
from data.utils import load_complement_graph


class DCATTrainer(object):
    """
    Examples:
        ```
        trainer = DCATTrainer(args)
        trainer.train()
        ...
        trainer.reset_args(args)
        trainer.train()
        ```

    Notes:
        This class cannot handle randomness in dataloading.
    """
    def __init__(self, args):
        self.args: argparse.Namespace
        self.n_splits: int  # The number of data splits
        self.split_type: int
        self.device: torch.device
        self._set_args(args)
        self._set_env(args)
        self.data = load_data(args.dataset, DATA_PATH, args.complement, True, load_tmp=args.load_tmp)

    def _set_args(self, args):
        assert args.split_type in ['fix', 'random', 'random_semi', 'stratified']
        assert isinstance(args.n_splits, int) and args.n_splits > 0
        self.args = args
        self.n_splits = args.n_splits  # The number of data splits
        self.split_type = args.split_type

    def _set_env(self, args):
        if args.seed >= 0:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
        # else:  # reset random seed
        #     np.random.seed()
        #     torch.random.set_rng_state(torch.random.get_rng_state())
        #
        self.device = choose_device(args.gpu)

    def reset_args(self, args):
        """Reset args and envs; train the model without reloading the dataset."""
        assert args.dataset == self.args.dataset
        if args.complement != self.args.complement:
            self.data.complementary_graph = load_complement_graph(
                self.data.raw_adj, args.complement)
        self._set_args(args)
        self._set_env(args)

    def train(self):
        if self.n_splits == 1:
            train_acc, val_acc, test_acc, nc_acc, nc_nmi, nc_cs = self._train_once()
            print("""Total Results:\n{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}""".format(
                    nc_acc, nc_nmi, nc_cs, train_acc, val_acc, test_acc))
        else:
            acc_list = []
            if self.split_type == 'fix':
                # For this kind, just run the same splits multiple times.
                for _ in range(self.n_splits):
                    accs = self._train_once()
                    acc_list.append(accs)
            elif self.split_type.startswith('random'):
                _semi = False
                if self.split_type == 'random_semi':
                    _semi = True
                cnt = 0
                for masks in load_random_splits(self.data.data_path, semi=_semi):
                    train_mask, val_mask, test_mask = masks
                    self.data._g.ndata['train_mask'] = train_mask
                    self.data._g.ndata['val_mask']   = val_mask
                    self.data._g.ndata['test_mask']  = test_mask
                    accs = self._train_once()
                    acc_list.append(accs)
                    cnt += 1
                    if cnt >= self.n_splits:
                        break
            else:
                raise NotImplementedError('No implementation for stratified sampling')
            train_acc = np.array([acc[0] for acc in acc_list])
            val_acc = np.array([acc[1] for acc in acc_list])
            test_acc = np.array([acc[2] for acc in acc_list])
            nc_acc = np.array([acc[3] for acc in acc_list])
            nc_nmi = np.array([acc[4] for acc in acc_list])
            nc_cs = np.array([acc[5] for acc in acc_list])
            print()
            print("Total Clustering ACC\t{:.4f} ({:.4f})".format(nc_acc.mean(), nc_acc.std()))
            print("Total Clustering NMI\t{:.4f} ({:.4f})".format(nc_nmi.mean(), nc_nmi.std()))
            print("Total Clustering CS\t{:.4f} ({:.4f})".format(nc_cs.mean(), nc_cs.std()))
            print("Total Train Accuracy\t{:.4f} ({:.4f})".format(train_acc.mean(), train_acc.std()))
            print("Total Val Accuracy\t{:.4f} ({:.4f})".format(val_acc.mean(), val_acc.std()))
            print("Total Test Accuracy\t{:.4f} ({:.4f})".format(test_acc.mean(), val_acc.std()))
            print("""Total Results:\n{:.4f} ({:.4f}),{:.4f} ({:.4f}),{:.4f} ({:.4f}),{:.4f} ({:.4f}),{:.4f} ({:.4f}),{:.4f} ({:.4f})""".format(
                nc_acc.mean(), nc_acc.std(),
                nc_nmi.mean(), nc_nmi.std(),
                nc_cs.mean(), nc_cs.std(),
                train_acc.mean(), train_acc.std(),
                val_acc.mean(), val_acc.std(),
                test_acc.mean(), val_acc.std()
            ))
        print('\n\n')

    def _train_once(self):
        g: dgl.DGLGraph = self.data[0]
        cg: dgl.DGLGraph = self.data.complementary_graph
        g = g.to(self.device)
        if isinstance(cg, dgl.DGLGraph):
            cg = cg.to(self.device)

        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        num_feats = features.shape[1]
        n_classes = self.data.num_labels
        n_edges = g.num_edges()

        # create model
        activation_fn = load_activation_fn(self.args.activation, self.args.negative_slope)
        heads = ([self.args.num_heads] * self.args.num_layers) + [self.args.num_out_heads]
        model = load_model(self.args.model,
                           self.args.num_layers,
                           num_feats,
                           self.args.num_hidden,
                           n_classes,
                           heads,
                           activation_fn,
                           self.args.in_drop,
                           self.args.attn_drop,
                           self.args.negative_slope,
                           self.args.residual,
                           reduced_dim=self.args.reduced_dim,
                           reg_last=self.args.reg_last,
                           postprocess=self.args.post_process,
                           # cauchy_c=self.args.cauchy_c,
                           beta=self.args.beta,
                           gamma=self.args.gamma)
        print(model)
        # Also print # parameters in the model
        total_params = sum(p.numel() for p in model.parameters())
        print('Model parameters: {}'.format(total_params))

        if self.args.early_stop:
            stopper = EarlyStopper(patience=self.args.es_patience, in_memory=True)
        loss_fcn = torch.nn.CrossEntropyLoss()
        learning_rate = self.args.lr
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=self.args.weight_decay)
        if self.args.lr_decay:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        model.to(self.device)
        beta = self.args.beta

        dur = []
        for epoch in range(self.args.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            if self.args.model != 'gat':
                logits, loss0 = model(g, features, cg)
                loss = loss_fcn(logits[train_mask], labels[train_mask]) * (1 - beta) + loss0 * beta
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
                val_acc = evaluate(model, labels, val_mask, g, features)
                if self.args.early_stop:
                    if stopper.step(val_acc, model):
                        break
                val_loss = loss_fcn(logits[val_mask], labels[val_mask])
                if self.args.lr_decay:
                    scheduler.step(val_loss)

            with warnings.catch_warnings():
                # Handle the warnings during throughput computation, which is useless.
                warnings.simplefilter("ignore")
                if self.args.model != 'gat':
                    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                          " ValLoss {:.4f} | ValAcc {:.4f} | Reg {:.4f} | ETputs(KTEPS) {:.2f}".
                          format(epoch, np.mean(dur), loss.item(), train_acc,
                                 val_loss.item(), val_acc, loss0.item(), n_edges / np.mean(dur) / 1000))
                else:
                    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                          " ValLoss {:.4f} | ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
                          format(epoch, np.mean(dur), loss.item(), train_acc,
                                 val_loss.item(), val_acc, n_edges / np.mean(dur) / 1000))
        print()

        if self.args.early_stop:
            model.load_state_dict(stopper.load_checkpoint())

        with torch.no_grad():
            model.eval()
            logits = model(g, features)
            nc_acc, nc_nmi, nc_cs = cluster_metrics(logits, labels)
            print("Node Clustering ACC\t{:.4f}".format(nc_acc))
            print("Node Clustering NMI\t{:.4f}".format(nc_nmi))
            print("Node Clustering CS\t{:.4f}".format(nc_cs))
            # Node classification
            train_acc = accuracy(logits[train_mask], labels[train_mask])
            print("Train Accuracy\t{:.4f}".format(train_acc))
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            print("Val Accuracy\t{:.4f}".format(val_acc))
            acc = accuracy(logits[test_mask], labels[test_mask])
            print("Test Accuracy\t{:.4f}".format(acc))

            # if self.args.dump_attn:
            #     self.dump_attention_score(model, g, features, self.args.dump_name)

            # if self.args.dump_hid_rep:
            #     self.dump_hidden_representation(model, g, features, self.args.dump_name)
            #     # self.dump_labels(labels, self.args.dump_name + '_labels')
            # # self.dump_loss(ols, self.args.dump_name)

        log_end()
        return (train_acc, val_acc, acc, nc_acc, nc_nmi, nc_cs)

    @staticmethod
    def dump_attention_score(model, g, features, filename):
        # """Dump attention scores from hidden layers."""
        # with torch.no_grad():
        #     model.eval()
        #     logits, attns = model(g, features, output_attn=True)
        #     for a in attns:
        #         print(a.shape)
        #     attns = [attn.cpu().numpy() for attn in attns]
        #     print(filename)
        #     np.savez_compressed(filename, *attns)
        #     print('Dump Finished')
        with torch.no_grad():
            with g.local_scope():
                model.eval()
                logits, attns = model(g, features, output_attn=True)
                for a in attns:
                    print(a.shape)
                assert len(attns) == 1
                g.edata['a'] = attns[0]
                dgl.save_graphs(filename, g)
                print('Dump Finished')

    @staticmethod
    def dump_hidden_representation(model, g, features, filename):
        with torch.no_grad():
            if isinstance(model, ProcessedGATWithLoss) or isinstance(model, ProcessedGAT):
                h = model.in_process(features)
            else:
                h = features
            for l in range(model.num_layers):
                h = model.gat_layers[l](g, h).flatten(1)
            np.savez_compressed(filename, a=h.detach().cpu().numpy())

    @staticmethod
    def dump_loss(losses, filename):
        losses = [loss.detach().cpu().numpy() for loss in losses]
        np.savez_compressed(filename, np.array(losses))

    @staticmethod
    def dump_labels(labels, filename):
        np.savez_compressed(filename, a=labels.detach().cpu().numpy())

    @staticmethod
    def get_args(arg_list=None):
        parser = argparse.ArgumentParser(description='GAT')
        register_data_args(parser)
        parser.add_argument("--load-tmp", action='store_true', default=False,
                            help='load `tmp` dataset split.')
        parser.add_argument("--script", type=str, default='train',
                            choices=['train', 'gen_index', 'save_mf'])

        parser.add_argument('--n-splits', type=int, default=1,
                            help="default for using only one split.")
        parser.add_argument('--split-type', type=str, default='fix',
                            choices=['fix', 'random', 'random_semi', 'stratified'])

        parser.add_argument("--seed", type=int, default=42,
                            help='A negative value indicates no fixed seed.')
        parser.add_argument("--gpu", type=int, default=-1,
                            help="which GPU to use. Set -1 to use CPU.")
        parser.add_argument("--epochs", type=int, default=200,
                            help="number of training epochs")
        parser.add_argument("--lr", type=float, default=0.005,
                            help="learning rate")
        parser.add_argument('--weight-decay', type=float, default=5e-4,
                            help="weight decay")
        parser.add_argument('--early-stop', action='store_true', default=False,
                            help="indicates whether to use early stop or not")
        parser.add_argument('--es-patience', type=int, default=100,
                            help='patience of early-stopping')
        parser.add_argument('--lr-decay', action="store_true", default=False,
                            help="use learning rate decay or not")
        parser.add_argument('--step-size', type=int, default=50,
                            help="Step size for learning rate decay")
        parser.add_argument("--in-drop", type=float, default=.6,
                            help="input feature dropout")
        parser.add_argument("--attn-drop", type=float, default=.6,
                            help="attention dropout")

        parser.add_argument('--reduced-dim', type=int, default=-1,
                            help='A linear layer for dim reduction, <=0 for no reduction.')
        parser.add_argument('--post-process', action='store_true', default=False)
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
        parser.add_argument('--activation', type=str, default='elu',
                            choices=['leaky_relu', 'elu', 'selu', 'celu', 'sin', 'relu'],
                            help="activation function")
        parser.add_argument('--negative-slope', type=float, default=0.2,
                            help="the negative slope of leaky relu for attention computation")
        parser.add_argument('--beta', type=float, default=0.1,
                            help="")
        parser.add_argument('--gamma', type=float, default=1,
                            help="the weight of orthogonal loss")
        parser.add_argument('--model', type=str, default='gat',
                            help='which model to use')
        parser.add_argument('--complement', type=str, default=None,
                            choices=[None, 'Laplacian', 'lap-sym-norm', 'lap-rw-norm',
                                     'adjacent', 'plain-adj'])
        parser.add_argument('--reg-last', action='store_true', default=False,
                            help='whether to use graph attn regularization in the last layer.')
        parser.add_argument('--dump-attn', action='store_true', default=False)
        parser.add_argument('--dump-hid-rep', action='store_true', default=False)
        parser.add_argument('--dump-name', type=str, default='')

        if arg_list is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(arg_list)

        log_time()
        print(args)
        return args


def main():
    args = DCATTrainer.get_args(['--dataset', 'wiki', '--complement', 'lap-sym-norm'])
    trainer = DCATTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
