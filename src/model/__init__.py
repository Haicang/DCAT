from .model import *
from .gt_model import *


def load_model(model_name: str, *params, reduced_dim=0, reg_last=False, postprocess=False, **kwargs):
    if reduced_dim <= 0:
        if model_name in ['gat']:
            return GAT(*params)
        if model_name in reg_list:
            return GATWithLoss(*params, reg=model_name, reg_last=reg_last, **kwargs)
        else:
            raise NotImplementedError
    elif postprocess is not True:
        if model_name in ['gat']:
            return ProcessedGAT(*params, reduced_dim=reduced_dim)
        if model_name in reg_list:
            return ProcessedGATWithLoss(*params,
                                        reg=model_name,
                                        reduced_dim=reduced_dim,
                                        **kwargs)
        else:
            raise NotImplementedError
    else:
        if model_name in ['gat']:
            return TwoProcessedGAT(*params, reduced_dim=reduced_dim)
        if model_name in reg_list:
            return TwoProcessedGATWithLoss(*params,
                                           reg=model_name,
                                           reduced_dim=reduced_dim,
                                           **kwargs)
        else:
            raise NotImplementedError


def load_gt(model_name: str, in_dim, num_hid, num_classes, n_layers, n_heads, dropout, use_norm = True, **kwargs):
    if model_name in ['gt']:
        return HGT(in_dim, num_hid, num_classes, n_layers, n_heads, dropout, use_norm)
    else:
        return HGTWithLoss(in_dim, num_hid, num_classes, n_layers, n_heads, dropout, use_norm, model_name, **kwargs)
