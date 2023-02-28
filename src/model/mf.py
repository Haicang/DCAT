import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF
import torch
import dgl
import dgl.function as fn

from utils import *
from data.utils import process_self_circle


__all__ = [
    'compute_proximity_matrix',
    'matrix_factorization',
    'graph_mf',
    'static_mf_reg'
]


def compute_proximity_matrix(mx):
    """
        .. math::
            0.5 * A^2 + 0.5 * A
        Here A is a normalized adj matrix.

    Args:
        mx (sp.spmatrix) : unnormalized adj matrix

    Returns:
        sp.spmatrix : proximity matrix for MF
    """
    adj = normalize(mx)
    return (adj @ adj + adj) * 0.5


def matrix_factorization(mx, dim=128, random_state=None):
    """

    Args:
        mx (sp.spmatrix):
        dim (int): dimension of output factorization
        random_state: random state of MF, for reproducible issue

    Returns:
        np.ndarray
    """
    mf = NMF(dim, random_state=random_state, verbose=2)
    vec = mf.fit_transform(mx)
    return vec


def graph_mf(adj, dim=128, random_state=None) -> np.ndarray:
    pm = compute_proximity_matrix(adj)
    return matrix_factorization(pm, dim=dim, random_state=random_state)


def static_mf_reg(graph, dim=128, random_state=None, load_path=None):
    """

    Args:
        graph (dgl.DGLGraph): DGLGraph built on adj matrix
        dim (int):
        random_state:

    Returns:
        dgl.DGLGraph:
    """
    adj = graph.adjacency_matrix(scipy_fmt='coo')
    if load_path is None:
        feat = graph_mf(adj, dim, random_state)
        assert feat.shape[0] == graph.num_nodes(), "Do NOT match the #nodes."
    else:
        feat = np.load(load_path)
    graph = process_self_circle(graph)
    # TODO: handle the self circle in the same way as in dataset class.

    default_dtype = torch.get_default_dtype()
    feat = torch.tensor(feat, dtype=default_dtype)
    with graph.local_scope():
        graph.ndata['feat'] = feat
        graph.apply_edges(fn.u_dot_v('feat', 'feat', 'w'))
        edata = graph.edata['w']
        edata = torch.sum(edata, dim=1)
        graph.edata['w'] = edata
        wadj = dgl_graph_to_weighted_adj(graph, 'w')
    g = dgl.from_scipy(wadj, eweight_name='w')
    return g
