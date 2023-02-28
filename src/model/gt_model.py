import torch.nn as nn
import torch.nn.functional as F
from .gt_layers import HGTLayer, HGTLayerWithLoss


__all__ = [
    'HGT',
    'HGTWithLoss',
]


class HGT(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, n_layers, n_heads, dropout, use_norm = True):
        super(HGT, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_all_hid = n_hid * n_heads
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.Linear(n_inp,   self.n_all_hid)
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid * n_heads, n_hid * n_heads, n_heads, dropout=dropout, use_norm = use_norm))
        self.out = nn.Linear(self.n_all_hid, n_out)

    def forward(self, G, h=None):
        if h is None:
            h = G.ndata['feat']
        h = F.gelu(self.adapt_ws(h))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return self.out(h)


class HGTWithLoss(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, n_layers, n_heads, dropout, use_norm=True, reg=None, **kwargs):
        super(HGTWithLoss, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_all_hid = n_hid * n_heads
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.Linear(n_inp, self.n_all_hid)
        self.reg = reg
        assert self.reg is not None
        for _ in range(n_layers):
            self.gcs.append(HGTLayerWithLoss(n_hid * n_heads, n_hid * n_heads, n_heads, dropout=dropout, use_norm=use_norm, reg=self.reg, **kwargs))
        self.out = nn.Linear(self.n_all_hid, n_out)

    def forward(self, G, h, *args):
        if self.training:
            h = F.gelu(self.adapt_ws(h))
            L = 0
            for i in range(self.n_layers):
                h, loss = self.gcs[i](G, h, *args)
                L += loss
            return self.out(h), L / self.n_layers
        else:
            h = F.gelu(self.adapt_ws(h))
            for i in range(self.n_layers):
                h = self.gcs[i](G, h)
            return self.out(h)