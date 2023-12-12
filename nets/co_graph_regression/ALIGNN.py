"""ALIGNN(Atomistic Line Graph Neural Network): dgl implementation.
doi: 10.1038/s41524-021-00650-1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from nets.co_graph_regression.utils import RBFExpansion


class ScaleNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean_square = (x ** 2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.scala
        return x

class EdgeGatedGraphConv(nn.Module):
    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,) -> torch.Tensor:
        g = g.local_var()
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["feat"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("feat")

        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y
        return x, y


class ALIGNNNET(nn.Module):
    def __init__(self, net_params):

        super().__init__()
        num_atom_type = net_params['num_atom_type']  # 节点维度 5
        num_bond_type = net_params['num_bond_type']  # 边的维度 1
        num_theta_type = net_params['num_theta_type']  # 角的维度 1
        edge_input_dim = net_params['edge_input_dim']  # 80
        t_input_dim = net_params['t_input_dim']  # 40
        embedding_dim = net_params['embedding_dim'] # 64
        embedding_features = net_params['embedding_features']
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']  # 256
        out_dim = net_params['out_dim']  # 64
        in_feat_dropout = net_params['in_feat_dropout']  # 0.0
        dropout = net_params['dropout']  # 0.03
        self.n_layers = net_params['L']  # 10
        self.k_layers = net_params['K']  # 10
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

        if self.soap:
            soap_enc_dim = net_params['soap_enc_dim']
            self.embedding_soap = nn.Sequential(nn.Linear(soap_enc_dim, embedding_dim//2),)
            self.embedding_h = nn.Sequential(nn.Linear(num_atom_type, embedding_dim//2),)
        else:
            self.embedding_h = nn.Sequential(nn.Linear(num_atom_type, embedding_dim),)

        self.embedding_lh = nn.Linear(num_bond_type, hidden_dim)
        self.embedding_e = nn.Linear(num_bond_type, hidden_dim)
        self.embedding_le = nn.Linear(num_theta_type, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins = edge_input_dim),
            nn.Linear(edge_input_dim, embedding_features),
            nn.Softplus(),
            nn.Linear(embedding_features, in_dim))
        self.abf = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1, bins =  t_input_dim),
            nn.Linear(t_input_dim, embedding_features),
            nn.Softplus(),
            nn.Linear(embedding_features, in_dim))

        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(out_dim, out_dim) for idx in range(self.n_layers)])
        self.lgcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(out_dim, out_dim) for idx in range(self.n_layers)])

        self.gcn_end_layers = nn.ModuleList(
            [EdgeGatedGraphConv(out_dim, out_dim) for idx in range(self.k_layers)])

        self.fc = nn.Sequential(nn.Linear(out_dim, fc_features), nn.SiLU(), nn.Linear(fc_features, 1))


    def forward(self, g, lg, h, e, lh, le, _=None, h_soap=None):
        h = self.embedding_h(h)
        lh = self.rbf(lh)

        if not self.edge_feat: # edge feature set to 1
            le = torch.ones(e.size(0),1).to(self.device)

        le = self.abf(le)
        h = self.in_feat_dropout(h)
        if self.soap:
            h_soap = self.embedding_soap(h_soap)
            h = torch.cat([h, h_soap], dim=1)

        for gcn,lgcn in zip(self.gcn_layers, self.lgcn_layers):
            lh, le = lgcn(lg, lh, le)
            h, lh = gcn(g, h, lh)

        for gcn in self.gcn_end_layers:
            h, lh = gcn(g, h, lh)

        g.ndata['h'] = h
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

