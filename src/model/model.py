"""
Residual is not implemented in current version.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *


__all__ = [
    'load_activation_fn',
    'GAT',
    'GATWithLoss',
    'ProcessedGAT',
    'TwoProcessedGAT',
    'ProcessedGATWithLoss',
    'TwoProcessedGATWithLoss',
    'reg_list'
]


#################################
#  Choose Activation Functions  #
#################################

def load_activation_fn(fn_name, *args):
    if fn_name == 'leaky_relu':
        return partial(F.leaky_relu, negative_slope=args[0])
    elif fn_name == 'elu':
        return F.elu
    elif fn_name == 'relu':
        return F.relu
    elif fn_name == 'selu':
        return F.selu
    elif fn_name == 'celu':
        return partial(F.celu, alpha=args[0])
    elif fn_name == 'sin':
        return torch.sin
    else:
        raise NotImplementedError


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers  # number of hidden layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GraphAttnLayer(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttnLayer(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GraphAttnLayer(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs, output_attn=False, out_rep=False):
        h = inputs
        if output_attn:
            attns = []
            for l in range(self.num_layers):
                h, attn = self.gat_layers[l](g, h, output_attn=output_attn)
                h = h.flatten(1)
                attns.append(attn)
            # output projection
            logits, attn = self.gat_layers[-1](g, h, output_attn=output_attn)
            logits = logits.mean(1)
            attns.append(attn)
            return logits, attns
        elif out_rep:
            reps = []
            for l in range(self.num_layers):
                h, rep = self.gat_layers[l](g, h, out_rep=out_rep)
                h = h.flatten(1)
                reps.append(rep)
            # output projection
            logits, rep = self.gat_layers[-1](g, h, out_rep=out_rep)
            logits = logits.mean(1)
            reps.append(rep)
            return logits, reps
        else:
            for l in range(self.num_layers):
                h = self.gat_layers[l](g, h).flatten(1)
                # output projection
            logits = self.gat_layers[-1](g, h).mean(1)
            return logits


class GATWithLoss(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 reg,
                 reg_last=False,
                 **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.reg = reg
        self.reg_last = reg_last
        # input projection (no residual)
        self.gat_layers.append(GraphAttnLayerWithLoss(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, reg=reg, **kwargs))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttnLayerWithLoss(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, reg=reg, **kwargs))
        # output projection
        if self.reg_last is False:
            self.gat_layers.append(GraphAttnLayer(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GraphAttnLayerWithLoss(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None, reg=reg))

    def forward(self, g, inputs, *args, output_attn=False, out_rep=False):
        h = inputs
        if self.training:
            L = 0
            for l in range(self.num_layers):
                h, loss = self.gat_layers[l](g, h, *args)
                h = h.flatten(1)
                L += loss
                # output projection
            if self.reg_last:
                h, loss = self.gat_layers[-1](g, h, *args)
                logits = h.mean(1)
                L += loss
            else:
                logits = self.gat_layers[-1](g, h).mean(1)
            # print(logits.shape)
            return logits, L
        elif output_attn:
            attns = []
            for l in range(self.num_layers):
                h, attn = self.gat_layers[l](g, h, output_attn=output_attn)
                h = h.flatten(1)
                attns.append(attn)
            logits, attn = self.gat_layers[-1](g, h, output_attn=output_attn)
            logits = logits.mean(1)
            # attns.append(attn)
            return logits, attns
        elif out_rep:
            reps = []
            for l in range(self.num_layers):
                h, rep = self.gat_layers[l](g, h, out_rep=out_rep)
                h = h.flatten(1)
                reps.append(rep)
            # output projection
            logits, rep = self.gat_layers[-1](g, h, out_rep=out_rep)
            logits = logits.mean(1)
            reps.append(rep)
            return logits, reps
        else:
            for l in range(self.num_layers):
                h = self.gat_layers[l](g, h).flatten(1)
            logits = self.gat_layers[-1](g, h).mean(1)
            return logits


#####################################
#  Model with preprocessing layers  #
#####################################

# In order to keep the same APIs, I can only choose to use inheritance.

class ProcessedGAT(GAT):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 reduced_dim):
        super().__init__(num_layers, reduced_dim, num_hidden, num_classes, heads,
                         activation, feat_drop, attn_drop, negative_slope, residual)
        self.in_process = nn.Linear(in_dim, reduced_dim, bias=False)

    def forward(self, g, inputs, output_attn=False):
        inputs = self.in_process(inputs)
        return super().forward(g, inputs, output_attn=output_attn)


class TwoProcessedGAT(GAT):
    """Pre-processing + post-processing"""
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 reduced_dim):
        super().__init__(num_layers, reduced_dim, num_hidden, num_hidden, heads,
                         activation, feat_drop, attn_drop, negative_slope, residual)
        self.in_process = nn.Linear(in_dim, reduced_dim, bias=False)
        self.out_process = nn.Linear(num_hidden, num_classes, bias=False)

    def forward(self, g, inputs, output_attn=False):
        x = self.in_process(inputs)
        x = super().forward(g, x, output_attn)
        x = self.activation(x)
        x = self.out_process(x)
        return x


class ProcessedGATWithLoss(GATWithLoss):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 reg,
                 reduced_dim,
                 reg_last=False,
                 **kwargs):
        super().__init__(num_layers, reduced_dim, num_hidden, num_classes, heads, activation,
                         feat_drop, attn_drop, negative_slope, residual, reg, reg_last, **kwargs)
        self.in_process = nn.Linear(in_dim, reduced_dim, bias=False)

    def forward(self, g, inputs, *args, output_attn=False):
        inputs = self.in_process(inputs)
        return super().forward(g, inputs, *args, output_attn=output_attn)


class TwoProcessedGATWithLoss(GATWithLoss):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 reg,
                 reduced_dim,
                 reg_last=False,
                 **kwargs):
        super().__init__(num_layers, reduced_dim, num_hidden, num_hidden, heads, activation,
                         feat_drop, attn_drop, negative_slope, residual, reg, reg_last, **kwargs)
        self.in_process = nn.Linear(in_dim, reduced_dim, bias=False)
        self.out_process = nn.Linear(num_hidden, num_classes, bias=False)

    def forward(self, g, inputs, *args):
        x = self.in_process(inputs)
        x = super().forward(g, x, *args)
        x = self.activation(x)
        logits = self.out_process(x)
        return logits
