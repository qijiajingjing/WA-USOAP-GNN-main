"""CGCNN: dgl implementation."""
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from nets.co_graph_regression.utils import RBFExpansion

class CGCNNConv(nn.Module):
    """Xie and Grossman graph convolution function.
    10.1103/PhysRevLett.120.145301
    """
    def __init__(
        self,
        node_features: int = 64,
        edge_features: int = 32,
        return_messages: bool = False,
    ):
        """Initialize torch modules for CGCNNConv layer."""
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.return_messages = return_messages

        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.linear_src = nn.Linear(node_features, 2 * node_features)
        self.linear_dst = nn.Linear(node_features, 2 * node_features)
        self.linear_edge = nn.Linear(edge_features, 2 * node_features)
        self.bn_message = nn.BatchNorm1d(2 * node_features)

        # final batchnorm
        self.bn = nn.BatchNorm1d(node_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:

        g = g.local_var()


        g.ndata["h_src"] = self.linear_src(node_feats)
        g.ndata["h_dst"] = self.linear_dst(node_feats)
        g.apply_edges(fn.u_add_v("h_src", "h_dst", "h_nodes"))
        m = g.edata.pop("h_nodes") + self.linear_edge(edge_feats)
        m = self.bn_message(m)

        # split messages into W_f and W_s terms
        # multiply output of atom interaction net and edge attention net
        # i.e. compute the term inside the summation in eq 5
        # σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        h_f, h_s = torch.chunk(m, 2, dim=1)
        m = torch.sigmoid(h_f) * F.softplus(h_s)
        g.edata["m"] = m

        # apply the convolution term in eq. 5 (without residual connection)
        # storing the results in edge features `h`
        g.update_all(
            message_func=fn.copy_e("m", "z"), reduce_func=fn.sum("z", "h"),
        )

        # final batchnorm
        h = self.bn(g.ndata.pop("h"))

        # residual connection plus nonlinearity
        out = F.softplus(node_feats + h)

        if self.return_messages:
            return out, m

        return out


class CGCNNNET(nn.Module):
    """CGCNN dgl implementation."""

    # def __init__(self, config: CGCNNConfig = CGCNNConfig(name="cgcnn")):
    def __init__(self, net_params):
        """Set up CGCNN modules."""
        super().__init__()
        num_atom_type = net_params['num_atom_type']  # 节点维度 5
        num_bond_type = net_params['num_bond_type']  # 边的维度 1
        edge_input_dim = net_params['edge_input_dim']  # 80
        embedding_dim = net_params['embedding_dim']  # 64
        embedding_features = net_params['embedding_features']
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']  # 256
        out_dim = net_params['out_dim']  # 64
        in_feat_dropout = net_params['in_feat_dropout']  # 0.0
        dropout = net_params['dropout']  # 0.0
        self.n_layers = net_params['L']  # 10
        fc_features = net_params['fc_features']

        self.net_params = net_params
        self.readout = net_params['readout']  # 'mean'
        self.layer_norm = net_params['layer_norm']  # ture
        self.batch_norm = net_params['batch_norm']  # false
        self.residual = net_params['residual']  # true
        self.edge_feat = net_params['edge_feat']  # true
        self.device = net_params['device']  # 3
        self.soap = net_params['h_soap']  # true
        self.h_cat = net_params['h_cat']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        if self.soap:
            soap_enc_dim = net_params['soap_enc_dim']
            self.embedding_soap = nn.Sequential(nn.Linear(soap_enc_dim, embedding_dim//2),)
            self.embedding_h = nn.Sequential(nn.Linear(num_atom_type, embedding_dim//2),)
        else:
            self.embedding_h = nn.Sequential(nn.Linear(num_atom_type, embedding_dim),)

        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=edge_input_dim),
            nn.Linear(edge_input_dim, embedding_features),
            nn.Softplus(),
            nn.Linear(embedding_features, in_dim))
        self.atom_embedding = nn.Linear(num_atom_type, embedding_dim)

        self.conv_layers = nn.ModuleList(
            [
                CGCNNConv(embedding_dim, in_dim)
                for _ in range(self.n_layers)
            ]
        )

        self.fc = nn.Sequential(nn.Linear(out_dim, fc_features), nn.SiLU(), nn.Linear(fc_features, 1))

    def forward(self, g, lg, h, e, lh, le=None, _=None, h_soap=None):
        """CGCNN function mapping graph to outputs."""
        h = self.embedding_h(h)
        lh = self.rbf(lh)
        h = self.in_feat_dropout(h)
        if self.soap:
            h_soap = self.embedding_soap(h_soap)
            h = torch.cat([h, h_soap], dim=1)

        # CGCNN-Conv block: update node features
        for conv_layer in self.conv_layers:
            h = conv_layer(g, h, lh)
        g.ndata['h'] = h
        # crystal-level readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
        return self.fc(hg)
    def loss(self, scores, targets):
        loss = nn.MSELoss()(scores, targets)
        return loss


