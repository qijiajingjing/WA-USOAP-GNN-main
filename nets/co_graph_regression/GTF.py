
import torch
import torch.nn as nn
import dgl
from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout
from nets.co_graph_regression.utils import RBFExpansion

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        embedding_dim = net_params['embedding_dim']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.soap = net_params['h_soap']
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins = 40),
            nn.Linear(40, embedding_dim),
            nn.Softplus(),
            nn.Linear(embedding_dim, embedding_dim))
        if self.soap:
            soap_enc_dim = net_params['soap_enc_dim']
            self.embedding_soap = nn.Sequential(nn.Linear(soap_enc_dim, embedding_dim//2),)
            self.embedding_h = nn.Sequential(nn.Linear(num_atom_type, embedding_dim//2),)
        else:
            self.embedding_h = nn.Sequential(nn.Linear(num_atom_type, embedding_dim),)

        if self.edge_feat:
            self.embedding_e = nn.Linear(1, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                           self.layer_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, lg, h, e, lh, le, _, h_soap=None):
        ''' (batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc) '''
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.soap:
            h_soap = self.embedding_soap(h_soap)
            h = torch.cat([h, h_soap], dim=1)
        e = self.rbf(lh)

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        print(scores.shape)
        print(targets.shape)
        loss = nn.MSELoss()(scores, targets)
        return loss



