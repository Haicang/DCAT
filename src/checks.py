"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

from train_split import *


class CheckTrainer(RGATTrainer):
    """Check the distribution of angles"""
    def __init__(self, args):
        super().__init__(args)
        assert self.n_splits == 1

    def train(self):
        assert self.n_splits == 1
        train_acc, val_acc, test_acc, nc_acc, nc_nmi, nc_cs = self._train_once()
        print("""Total Results:\n{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}""".format(
            nc_acc, nc_nmi, nc_cs, train_acc, val_acc, test_acc))

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
                           cauchy_c=self.args.cauchy_c,
                           beta=self.args.beta,
                           gamma=self.args.gamma)
        print(model)
        if self.args.early_stop:
            stopper = EarlyStopper(patience=self.args.es_patience, in_memory=True)
        loss_fcn = torch.nn.CrossEntropyLoss()
        learning_rate = self.args.lr
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=self.args.weight_decay)
        if self.args.lr_decay:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        model.to(self.device)

        dur = []
        # dump hidden representation before training
        model.eval()
        self.dump_hidden_representation(model, g, features, self.args.dump_name + '_init')
        model.train()
        for epoch in range(self.args.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # forward
            if self.args.model != 'gat':
                logits, loss0 = model(g, features, cg)
                loss = loss_fcn(logits[train_mask], labels[train_mask]) + \
                       loss0 * self.args.beta
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
                          format(epoch, np.mean(dur), (loss - loss0 * self.args.beta).item(), train_acc,
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

            # dump hidden representation after training
            self.dump_hidden_representation(model, g, features, self.args.dump_name + '_final')

        log_end()
        return (train_acc, val_acc, acc, nc_acc, nc_nmi, nc_cs)

    @staticmethod
    def dump_hidden_representation(model, g, features, filename):
        """Overwrite the function to dump only the first hidden layer."""
        with torch.no_grad():
            if isinstance(model, ProcessedGATWithLoss) or isinstance(model, ProcessedGAT):
                h = model.in_process(features)
            else:
                h = features
            # The attention head is on the 1st dimension (index starts from 0).
            h, rep = model.gat_layers[0](g, h, out_rep=True)
            np.savez_compressed(filename, a=rep.detach().cpu().numpy())


def main():
    args = RGATTrainer.get_args(['--dataset', 'wiki', '--complement', 'lap-sym-norm'])
    trainer = RGATTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
