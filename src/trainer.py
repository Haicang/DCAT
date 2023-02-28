import abc
import time
from argparse import Namespace
from typing import Union

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import dgl

from utils import *
from data.utils import sample_mask


__all__ = [
    'GradientTrainer',
    'LBFGSTrainer',
    'GNNsTrainer',
    'GNNsTrainerMultiTask',
]


class BaseTrainer(abc.ABC):
    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 epochs,
                 stopper=None,
                 acc_fn=accuracy,
                 log_fn=None,
                 verbose=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.stopper = stopper
        self.acc_fn = acc_fn
        self.log_fn = log_fn
        self.verbose = verbose

    @property
    def has_stopper(self):
        return self.stopper is not None

    @staticmethod
    def _check_inputs(feats) -> list:
        if not isinstance(feats, list):
            return [feats]
        return feats

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def score(self, *args):
        pass


class GradientTrainer(BaseTrainer):
    """Simple trainer for mlp like model."""
    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 epochs,
                 stopper=None,
                 acc_fn=accuracy,
                 log_fn=None,
                 verbose=True):
        super().__init__(model, criterion, optimizer, epochs, stopper, acc_fn, log_fn, verbose)

    def score(self, feats, labels):
        feats = self._check_inputs(feats)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(*feats)
            acc = self.acc_fn(outputs, labels)
            loss = self.criterion(outputs, labels)
        return acc, loss.item()

    def _simple_fit(self, feats, labels):
        """
        :param feats: torch.Tensor or list
        :param labels: torch.Tensor
        :return:
        """
        feats = self._check_inputs(feats)
        model: nn.Module = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        model.train()

        dur = []
        t_start = time.time()
        for epoch in range(self.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()

            optimizer.zero_grad()
            output = model(*feats)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            if self.verbose:
                if self.log_fn is None:
                    acc, train_loss = self.score(feats, labels)
                    print('Epoch {:5d} | Time(s) {:.4f} | Loss {:.4f} | Acc {:.4f}'.format(
                        epoch, np.mean(dur), train_loss, acc
                    ))
                else:
                    raise NotImplementedError
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

        self.model = model

    def _fit_with_val(self, feats, labels, vfeats, vlabels):
        feats = self._check_inputs(feats)
        vfeats = self._check_inputs(vfeats)
        model: nn.Module = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        stopper = self.stopper
        model.train()

        dur = []
        t_start = time.time()
        for epoch in range(self.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()

            optimizer.zero_grad()
            output = model(*feats)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            with torch.no_grad():
                train_acc, train_loss = self.score(feats, labels)
                val_acc, val_loss = self.score(vfeats, vlabels)
                if stopper is not None:
                    if stopper.step(val_acc, model):
                        break

            if self.verbose:
                if self.log_fn is None:
                    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                          " ValLoss {:.4f} | ValAcc {:.4f}".
                          format(epoch, np.mean(dur), train_loss, train_acc, val_loss, val_acc))
                else:
                    raise NotImplementedError
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

        if stopper is not None:
            model.load_state_dict(stopper.load_checkpoint())
        self.model = model

    def fit(self, feats, labels, vfeats=None, vlabels=None):
        """

        :param feats: Tensor or list of Tensors, training data
        :param labels: Tensor, training labels
        :param vfeats: Tensor or list of Tensors, validation data
        :param vlabels: Tensor, validation labels
        :return: None
        """
        assert ((vfeats is not None) and (vlabels is not None)) or \
               ((vfeats is None) and vlabels is None), 'Input error'
        if self.stopper is None:
            if vfeats is not None:
                self._fit_with_val(feats, labels, vfeats, vlabels)
            else:
                self._simple_fit(feats, labels)
        else:
            if vfeats is not None:
                self._fit_with_val(feats, labels, vfeats, vlabels)
            else:
                raise AssertionError('Early stopping requires validation set.')

    def log_results(self, feats: torch.Tensor, labels: torch.Tensor,
                    vfeats: torch.Tensor, vlabels: torch.Tensor,
                    tfeats: torch.Tensor, tlabels: torch.Tensor):
        feats = self._check_inputs(feats)
        vfeats = self._check_inputs(vfeats)
        tfeats = self._check_inputs(tfeats)
        train_acc, train_loss = self.score(feats, labels)
        val_acc, val_loss = self.score(vfeats, vlabels)
        test_acc, test_loss = self.score(tfeats, tlabels)
        print('>>> Train Accuracy\t{:.4f};\tTrain Loss\t{:.4f}'.format(train_acc, train_loss))
        print('    Val Accuracy\t{:.4f};\tVal Loss\t{:.4f}'.format(val_acc, val_loss))
        print('    Test Accuracy\t{:.4f};\tTest Loss\t{:.4f}'.format(test_acc, test_loss))


class LBFGSTrainer(BaseTrainer):
    """
    Simple trainer for mlp like model using LBFGS.

    Warning: instances of this class can only be used for one time.

    `criterion` is the loss function.
    """
    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 epochs,
                 stopper=None,
                 acc_fn=accuracy,
                 log_fn=None,
                 verbose=True):
        super().__init__(model, criterion, optimizer, epochs, stopper, acc_fn, log_fn, verbose)

    def score(self, feats, labels):
        feats = self._check_inputs(feats)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(*feats)
            acc = self.acc_fn(outputs, labels)
            loss = self.criterion(outputs, labels)
        return acc, loss.item()

    def _simple_fit(self, feats, labels):
        """
        :param feats: torch.Tensor or list
        :param labels: torch.Tensor
        :return:
        """
        feats = self._check_inputs(feats)
        model: nn.Module = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        model.train()

        dur = []
        t_start = time.time()
        for epoch in range(self.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()

            def closure():
                optimizer.zero_grad()
                output = model(*feats)
                loss = criterion(output, labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            if epoch >= 3:
                dur.append(time.time() - t0)

            if self.verbose:
                if self.log_fn is None:
                    acc, train_loss = self.score(feats, labels)
                    print('Epoch {:5d} | Time(s) {:.4f} | Loss {:.4f} | Acc {:.4f}'.format(
                        epoch, np.mean(dur), train_loss, acc
                    ))
                else:
                    raise NotImplementedError
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

        self.model = model

    def _fit_with_val(self, feats, labels, vfeats, vlabels):
        feats = self._check_inputs(feats)
        vfeats = self._check_inputs(vfeats)
        model: nn.Module = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        stopper = self.stopper
        model.train()

        dur = []
        t_start = time.time()
        for epoch in range(self.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()

            def closure():
                optimizer.zero_grad()
                output = model(*feats)
                loss = criterion(output, labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            if epoch >= 3:
                dur.append(time.time() - t0)

            with torch.no_grad():
                train_acc, train_loss = self.score(feats, labels)
                val_acc, val_loss = self.score(vfeats, vlabels)
                if stopper is not None:
                    if stopper.step(val_acc, model):
                        break

            if self.verbose:
                if self.log_fn is None:
                    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                          " ValLoss {:.4f} | ValAcc {:.4f}".
                          format(epoch, np.mean(dur), train_loss, train_acc, val_loss, val_acc))
                else:
                    raise NotImplementedError
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

        if stopper is not None:
            model.load_state_dict(stopper.load_checkpoint())
        self.model = model

    def fit(self, feats, labels, vfeats=None, vlabels=None):
        """

        :param feats: Tensor or list of Tensors, training data
        :param labels: Tensor, training labels
        :param vfeats: Tensor or list of Tensors, validation data
        :param vlabels: Tensor, validation labels
        :return: None
        """

        assert ((vfeats is not None) and (vlabels is not None)) or \
               ((vfeats is None) and vlabels is None), 'Input error'
        if self.stopper is None:
            if vfeats is not None:
                self._fit_with_val(feats, labels, vfeats, vlabels)
            else:
                self._simple_fit(feats, labels)
        else:
            if vfeats is not None:
                self._fit_with_val(feats, labels, vfeats, vlabels)
            else:
                raise AssertionError('Early stopping requires validation set.')

    def log_results(self, feats: torch.Tensor, labels: torch.Tensor,
                    vfeats: torch.Tensor, vlabels: torch.Tensor,
                    tfeats: torch.Tensor, tlabels: torch.Tensor):
        feats = self._check_inputs(feats)
        vfeats = self._check_inputs(vfeats)
        tfeats = self._check_inputs(tfeats)
        train_acc, train_loss = self.score(feats, labels)
        val_acc, val_loss = self.score(vfeats, vlabels)
        test_acc, test_loss = self.score(tfeats, tlabels)
        print('>>> Train Accuracy\t{:.4f};\tTrain Loss\t{:.4f}'.format(train_acc, train_loss))
        print('    Val Accuracy\t{:.4f};\tVal Loss\t{:.4f}'.format(val_acc, val_loss))
        print('    Test Accuracy\t{:.4f};\tTest Loss\t{:.4f}'.format(test_acc, test_loss))


class GNNsTrainer(BaseTrainer):
    """
    Trainer for GNN models. The transductive semi-supervised training of GNNs is different from
    full supervised training of mlp-like models.
    """
    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 epochs,
                 stopper=None,
                 acc_fn=accuracy,
                 log_fn=None,
                 verbose=True):
        super().__init__(model, criterion, optimizer, epochs, stopper, acc_fn, log_fn, verbose)
        self.validation = False

    @staticmethod
    def _check_mask_types(masks: list):
        for m in masks:
            assert isinstance(m, torch.Tensor), 'Type Error'
            assert m.dtype == torch.bool

    def score(self, feats: list, labels: torch.Tensor, mask: torch.Tensor):
        feats = self._check_inputs(feats)
        model = self.model
        criterion = self.criterion
        model.eval()

        with torch.no_grad():
            outputs = model(*feats)
            acc = self.acc_fn(outputs[mask], labels[mask])
            loss = criterion(outputs[mask], labels[mask])
        return acc, loss.item()

    def _simple_fit(self, feats, labels, train_mask):
        """

        Parameters
        ----------
        feats : torch.Tensor or list
        labels : torch.Tensor
        train_mask : torch.Tensor
            (of dtype `torch.int`)

        Returns
        -------
        out : None

        """
        feats = self._check_inputs(feats)
        model: nn.Module = self.model
        criterion = self.criterion
        optimizer = self.optimizer

        dur = []
        t_start = time.time()
        for epoch in range(self.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()

            optimizer.zero_grad()
            output = model(*feats)
            loss = criterion(output[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            if self.verbose:
                if self.log_fn is None:
                    acc, train_loss = self.score(feats, labels, train_mask)
                    print('Epoch {:5d} | Time(s) {:.4f} | Loss {:.4f} | Acc {:.4f}'.format(
                        epoch, np.mean(dur), train_loss, acc
                    ))
                else:
                    raise NotImplementedError
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

        self.model = model

    def _fit_with_val(self, feats, labels, train_mask, val_mask):
        feats = self._check_inputs(feats)
        model: nn.Module = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        stopper = self.stopper

        dur = []
        t_start = time.time()
        for epoch in range(self.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()

            optimizer.zero_grad()
            output = model(*feats)
            loss = criterion(output[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            with torch.no_grad():
                train_acc, train_loss = self.score(feats, labels, train_mask)
                val_acc, val_loss = self.score(feats, labels, val_mask)
                if stopper is not None:
                    if stopper.step(val_acc, model):
                        break

            if self.verbose:
                if self.log_fn is None:
                    print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                          " ValLoss {:.4f} | ValAcc {:.4f}".
                          format(epoch, np.mean(dur), train_loss, train_acc, val_loss, val_acc))
                else:
                    raise NotImplementedError
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

        if stopper is not None:
            model.load_state_dict(stopper.load_checkpoint())
        self.model = model

    def fit(self, feats: list, labels: torch.Tensor, masks: Union[torch.Tensor, list]):
        """

        Parameters
        ----------
        feats : list
        labels : torch.Tensor
        masks : torch.Tensor or list[torch.Tensor]

        Returns
        -------
        out : None
        """
        masks = self._check_inputs(masks)
        self._check_mask_types(masks)
        if len(masks) == 2:
            train_mask, val_mask = masks
            self.validation = True
            self._fit_with_val(feats, labels, train_mask, val_mask)
        elif len(masks) == 1:
            train_mask = masks[0]
            self.validation = False
            self._simple_fit(feats, labels, train_mask)
        else:
            raise TypeError('Something wrong with type of `masks`.')

    def log_results(self, feats: list, labels: torch.Tensor,
                    train_mask: torch.Tensor, val_mask: torch.Tensor,
                    test_mask: torch.Tensor):
        feats = self._check_inputs(feats)
        train_acc, train_loss = self.score(feats, labels, train_mask)
        val_acc, val_loss = self.score(feats, labels, val_mask)
        test_acc, test_loss = self.score(feats, labels, test_mask)
        print('>>> Train Accuracy\t{:.4f};\tTrain Loss\t{:.4f}'.format(train_acc, train_loss))
        print('    Val Accuracy\t{:.4f};\tVal Loss\t{:.4f}'.format(val_acc, val_loss))
        print('    Test Accuracy\t{:.4f};\tTest Loss\t{:.4f}'.format(test_acc, test_loss))


class GNNsTrainerMultiTask(GNNsTrainer):
    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 epochs,
                 stopper=None,
                 acc_fn=accuracy,
                 log_fn=None,
                 task=('classification', 'clustering'),
                 verbose=True):
        super().__init__(model, criterion, optimizer, epochs, stopper, acc_fn, log_fn, verbose)
        if isinstance(task, str):
            task = [task]
        elif isinstance(task, tuple):
            task = list(task)
        else:
            raise TypeError
        assert 'classification' in task
        self.task = task

    def log_results(self, feats: list, labels: torch.Tensor,
                    train_mask: torch.Tensor, val_mask: torch.Tensor,
                    test_mask: torch.Tensor):
        model = self.model
        with torch.no_grad():
            model.eval()
            logits = model(*feats)
            if 'clustering' in self.task:
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
            if 'clustering' in self.task:
                print("""Total Results:\n{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}""".format(
                    nc_acc, nc_nmi, nc_cs, train_acc, val_acc, acc))
