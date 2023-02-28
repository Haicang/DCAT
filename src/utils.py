import time
import datetime
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
import torch.optim
import dgl
import dgl.function as fn
from sklearn.metrics import normalized_mutual_info_score, completeness_score


__all__ = [
    'EarlyStopper',
    'accuracy',
    'cluster_metrics',
    'choose_device',
    'log_time',
    'log_end',
    'evaluate',
    'dgl_graph_to_weighted_adj',
    'normalize',
    'normalize_adj',
    'torch_sparse_tensor_to_sparse_matrix',
    'sparse_mx_to_torch_sparse_tensor',
    'laplacian',
    'torch_sparse_diag',
    'torch_sparse_identity',
    'torch_sparse_transpose',
    'check_grad',
    'normalize_graph',
    'graph_homophily'
]


class EarlyStopper(object):
    """
    Warnings : Please specify the ``fname`` if you want to run
        multiple models simultaneously.

    Parameters
    ----------
    patience : int
    in_memory : bool
        Save the state_dict of the model in memory or on disk
    fname : str
        If save state_dict on disk, it's the filename
    verbose : bool
        Whether to print verbose information or not.

    """
    def __init__(self, patience=10, in_memory=False,
                 fname='es_checkpoint.pt', verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.in_memory = in_memory
        self.state_dict = None
        self.write_name = fname
        self.verbose = verbose

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        if self.in_memory:
            # The `in_memory` version is slightly slower,
            # but it does NOT require a lot of writes to the disk.
            self.state_dict = deepcopy(model.state_dict())
        else:
            torch.save(model.state_dict(), self.write_name)

    def load_checkpoint(self):
        if self.in_memory:
            rst = self.state_dict
        else:
            rst = torch.load(self.write_name)
        return rst


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


@torch.no_grad()
def cluster_metrics(logits: torch.Tensor, labels: torch.Tensor):
    _, predicts = torch.max(logits, dim=1)
    correct = torch.sum(predicts == labels)
    nc_acc = correct.item() * 1.0 / labels.shape[0]
    predicts = predicts.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    nc_nmi = normalized_mutual_info_score(labels, predicts)
    nc_cs = completeness_score(labels, predicts)
    return nc_acc, nc_nmi, nc_cs


def choose_device(gpu_id: int) -> torch.device:
    assert isinstance(gpu_id, int)
    if not torch.cuda.is_available():
        gpu_id = -1
    if gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', gpu_id)
    return device


def log_time():
    """Print the server time, with timezone."""
    print(time.strftime("%Y-%m-%d %H:%M:%S (%Z)", time.localtime()))


def log_time_dev(zone=8):
    """
    Log local time
    Parameters
    ----------
    zone : int
        The timezone [-12, -11, ..., 0, ..., 12]

    Returns
    -------
    out : datetime.datetime
    """
    utc_dt = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    utc_local_dt = utc_dt.astimezone(datetime.timezone(datetime.timedelta(hours=zone)))
    print(utc_local_dt)
    return utc_local_dt


def log_end():
    print('='*10, ' END ', '='*10)


def evaluate(model: torch.nn.Module, labels: torch.Tensor, mask: torch.Tensor, *args) -> float:
    """
    :param args: the input of the model.
    """
    model.eval()
    with torch.no_grad():
        logits = model(*args)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


########################
## DGLGraph Operation ##
########################


@torch.no_grad()
def dgl_graph_to_weighted_adj(g, key):
    """

    Args:
        g (dgl.DGLGraph): `g.edata[key]` should be a scalar for each node.
        key (str): key name of the `edata` attribute.

    Returns:
        sp.spmatrix:
    """
    assert g.edata[key].squeeze().dim() == 1
    n_nodes = g.num_nodes()
    u, v = g.edges()
    eids = g.edge_ids(u, v)
    val = g.edata[key].squeeze()
    val = val[eids]
    u: torch.Tensor
    u = u.numpy()
    v = v.numpy()
    val = val.numpy().astype(np.float_)
    return sp.coo_matrix((val, (u, v)), shape=(n_nodes, n_nodes))


def normalize_graph(graph: dgl.DGLGraph, norm='symmetric'):
    with graph.local_scope():
        graph.edata['w'] = torch.ones(graph.num_edges(), device=graph.device)
        graph.update_all(fn.copy_e('w', 'm'),
                         fn.sum('m', 'd'))
        if norm == 'symmetric':
            graph.apply_edges(fn.u_mul_v('d', 'd', 'dm'))
            graph.edata['w'] = graph.edata['w'] / graph.edata['dm'].sqrt()
        elif norm == 'random_walk':
            graph.apply_edges(fn.e_div_u('w', 'd', 'w'))
        else:
            raise NotImplementedError
        return graph.edata['w']


######################
## Sparse Operation ##
######################


def normalize(mx):
    """
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : sp.spmatrix or np.ndarray

    Returns
    -------
    out : sp.spmatrix or np.ndarray
    """
    rowsum = np.array(mx.sum(1)).astype(np.float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def torch_sparse_tensor_to_sparse_matrix(spts: torch.sparse.Tensor) -> sp.coo_matrix:
    """Convert a COO sparse tensor in torch to a coo sparse matrix in scipy."""
    if spts.is_cuda:
        spts = spts.cpu()
    if not spts.is_coalesced():
        spts = spts.coalesce()
    shape = list(spts.shape)
    indices = spts.indices()
    values = spts.values()
    spmx = sp.coo_matrix((values.numpy(), (indices[0].numpy(), indices[1].numpy())), shape=shape)
    return spmx


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.sparse.Tensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def laplacian(spmx: sp.spmatrix, norm=None) -> sp.spmatrix:
    if norm is None:
        rowsum = np.array(spmx.sum(1)).flatten()
        degree = sp.diags(rowsum)
        rst = degree - spmx
    elif norm == 'sym':
        nadj = normalize_adj(spmx)
        rst = sp.identity(spmx.shape[0]) - nadj
    elif norm == 'rw':
        nadj = normalize(spmx)
        rst = sp.identity(spmx.shape[0]) - nadj
    else:
        raise NotImplementedError
    return rst.tocoo().astype(np.float32)


def torch_sparse_diag(vec: torch.Tensor) -> torch.sparse.Tensor:
    """Generate a sparse diagonal matrix from a dense vector."""
    sz = vec.size(0)
    indices = torch.arange(0, sz, device=vec.device)
    indices = torch.stack([indices, indices])
    return torch.sparse_coo_tensor(indices, vec, (sz, sz), dtype=vec.dtype, device=vec.device).coalesce()


def torch_sparse_identity(n: int, device=None, requires_grad=False) -> torch.sparse.Tensor:
    ones = torch.ones(n, device=device, requires_grad=requires_grad)
    return torch_sparse_diag(ones)


def torch_sparse_transpose(spts: torch.sparse.Tensor) -> torch.sparse.Tensor:
    spts = spts.coalesce()
    indices = spts.indices()
    new_spts = torch.sparse_coo_tensor(indices[[1, 0]], spts.values(), size=spts.size(),
                                       dtype=spts.dtype, device=spts.device)
    return new_spts.coalesce()


def check_grad(ts: torch.Tensor):
    with torch.no_grad():
        print(ts.grad.max(), ((ts.grad ** 2) ** 0.5).mean(), ts.grad.min())


############
#  Others  #
############

def graph_homophily(graph: dgl.DGLGraph, attr='label', has_self_circle=True):
    data = graph.ndata[attr].cpu()
    if has_self_circle:
        graph = dgl.remove_self_loop(graph)
    adj = graph.adj()
    adj = adj.coalesce()
    indices = adj.indices()
    count = torch.sum(data[indices[0]] == data[indices[1]]).item()
    return count / indices.shape[1]
