import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax

from .layers import _attn_s_smoothing_ortho, _attn_s_smoothing_ortho_bak


class HGTLayer(nn.Module):
    """
    `out_dim` is different from the general setting in GAT.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        # self.node_dict     = node_dict
        # self.edge_dict     = edge_dict
        self.num_types     = 1
        self.total_rel     = 1
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.Linear(in_dim,   out_dim)
        self.q_linears   = nn.Linear(in_dim,   out_dim)
        self.v_linears   = nn.Linear(in_dim,   out_dim)
        self.a_linears   = nn.Linear(out_dim,  out_dim)
        self.use_norm    = use_norm
        if use_norm:
            self.norms   = nn.LayerNorm(out_dim)
        else:
            self.norms   = nn.Identity()

        # for t in range(self.num_types):
        #     self.k_linears.append(nn.Linear(in_dim,   out_dim))
        #     self.q_linears.append(nn.Linear(in_dim,   out_dim))
        #     self.v_linears.append(nn.Linear(in_dim,   out_dim))
        #     self.a_linears.append(nn.Linear(out_dim,  out_dim))
        #     if use_norm:
        #         self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            k_linear = self.k_linears
            v_linear = self.v_linears
            q_linear = self.q_linears

            k = k_linear(h).view(-1, self.n_heads, self.d_k)
            v = v_linear(h).view(-1, self.n_heads, self.d_k)
            q = q_linear(h).view(-1, self.n_heads, self.d_k)

            relation_att = self.relation_att
            relation_pri = self.relation_pri
            relation_msg = self.relation_msg

            k = torch.einsum("bij,ijk->bik", k, relation_att)
            v = torch.einsum("bij,ijk->bik", v, relation_msg)

            G.srcdata['k'] = k
            G.dstdata['q'] = q
            G.srcdata['v'] = v

            G.apply_edges(fn.v_dot_u('q', 'k', 't'))
            attn_score = G.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
            attn_score = edge_softmax(G, attn_score, norm_by='dst')

            G.edata['t'] = attn_score.unsqueeze(-1)

            G.update_all(fn.u_mul_e('v', 't', 'm'), fn.sum('m', 't'))

            '''
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
            '''
            alpha = torch.sigmoid(self.skip)
            t = G.ndata['t'].view(-1, self.out_dim)
            trans_out = self.drop(self.a_linears(t))
            trans_out = trans_out * alpha + h * (1-alpha)
            new_h = self.norms(trans_out)
            return new_h


class HGTLayerWithLoss(HGTLayer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_heads,
                 dropout=0.2,
                 use_norm=False,
                 reg=None,
                 **kwargs):
        super().__init__(in_dim, out_dim, n_heads, dropout, use_norm)
        self.reg = reg
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']
        if reg == 'cgt':
            self.loss_fn = _attn_s_smoothing_ortho_bak
            self.bn = nn.BatchNorm1d(n_heads, momentum=False,
                                     affine=False, track_running_stats=False)
        else:
            raise NotImplementedError
        self.norm_model = ['attn_s_smooth_bn_ortho']

    def forward(self, G, h, *args, output_attn=False, out_rep=False):
        with G.local_scope():
            k_linear = self.k_linears
            v_linear = self.v_linears
            q_linear = self.q_linears

            k = k_linear(h).view(-1, self.n_heads, self.d_k)
            v = v_linear(h).view(-1, self.n_heads, self.d_k)
            q = q_linear(h).view(-1, self.n_heads, self.d_k)

            relation_att = self.relation_att
            relation_pri = self.relation_pri
            relation_msg = self.relation_msg

            k = torch.einsum("bij,ijk->bik", k, relation_att)
            v = torch.einsum("bij,ijk->bik", v, relation_msg)

            G.srcdata['k'] = k
            G.dstdata['q'] = q
            G.srcdata['v'] = v

            G.apply_edges(fn.v_dot_u('q', 'k', 't'))
            attn_score = G.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
            attn_score = edge_softmax(G, attn_score, norm_by='dst')

            G.edata['t'] = attn_score.unsqueeze(-1)

            G.update_all(fn.u_mul_e('v', 't', 'm'), fn.sum('m', 't'))

            '''
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
            '''
            alpha = torch.sigmoid(self.skip)
            t = G.ndata['t'].view(-1, self.out_dim)
            trans_out = self.drop(self.a_linears(t))
            trans_out = trans_out * alpha + h * (1-alpha)
            new_h = self.norms(trans_out)
            G.ndata['ft'] = new_h.reshape(-1, self.n_heads, self.d_k)

            if self.training:
                compl_graph = args[0]
                with compl_graph.local_scope():
                    loss = self.loss_fn(G, new_h, compl_graph, self.gamma)
                return new_h, loss
            elif output_attn:
                raise NotImplementedError
            elif out_rep:
                raise NotImplementedError

            return new_h
