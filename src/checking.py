import torch
import dgl


class CheckEdgeIds(object):
    """
    Check the edge ids
    """
    def __init__(self):
        self.eids = None
        self.flag = True

    def check_edge_ids(self, graph: dgl.DGLGraph):
        if self.flag is False:
            return

        with torch.no_grad():
            adj = graph.adj(ctx=graph.device)
            adj = adj.coalesce()
            indices = adj.indices()
            eids = graph.edge_ids(indices[0], indices[1])
            if self.eids is None:
                self.eids = eids
            else:
                self.flag = torch.equal(eids, self.eids)
            return eids
