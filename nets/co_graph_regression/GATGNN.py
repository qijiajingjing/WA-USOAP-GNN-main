"""GATGNN(Graph Convolutional Neural Networks with Global Attention): dgl implementation.
doi: 10.1039/D0CP01474E
"""



import torch
import torch.nn as nn

import dgl
from layers.gat_layer import GATLayer
from layers.mlp_readout_layer import MLPReadout


class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = 1
        self.device = net_params['device']
        self.soap = net_params['h_soap']
        if self.soap:
            soap_enc_dim = net_params['soap_enc_dim']
            self.embedding_soap = nn.Sequential(nn.Linear(soap_enc_dim, hidden_dim * num_heads//2),)
            self.embedding_h = nn.Sequential(nn.Linear(num_atom_type, hidden_dim * num_heads//2),)
        else:
            self.embedding_h = nn.Sequential(nn.Linear(num_atom_type, hidden_dim * num_heads),)
        # self.embedding_h = nn.Embedding(in_dim_node, hidden_dim * num_heads)  # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, self.n_classes)

    def forward(self, g, lg, h, e=None, lh=None, le=None, _=None, h_soap=None):
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.soap:
            h_soap = self.embedding_soap(h_soap)
            h = torch.cat([h, h_soap], dim=1)
        # GAT
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        # output
        h_out = self.MLP_layer(hg)

        return h_out

    def loss(self, scores, targets):
        print(scores.shape)
        print(targets.shape)
        loss = nn.MSELoss()(scores, targets)
        return loss