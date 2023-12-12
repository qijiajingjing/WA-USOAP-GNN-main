

from transformers.activations import ACT2FN
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from nets.co_graph_regression.utils import RBFExpansion
import numpy as np

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

class GAU(nn.Module):
    def __init__(self, net_params):
        # self, hidden_size=768, expansion_factor=2, s=128,
        # norm_type="layer_norm", eps=1e-5,
        # hidden_act="silu", max_position_embeddings=512,):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        expansion_factor = net_params['expansion_factor']
        self.s = net_params['s']
        self.e = int(hidden_dim * expansion_factor)
        self.uv = nn.Linear(hidden_dim, 2 * self.e + self.s)
        self.weight = nn.Parameter(torch.randn(2, self.s))
        self.bias = nn.Parameter(torch.zeros(2, self.s))
        self.o = nn.Linear(self.e, hidden_dim)
        norm_type = net_params["norm_type"]
        eps = net_params['eps']
        max_position_embeddings = net_params['max_position_embeddings']
        hidden_act = net_params['hidden_act']
        self.LayerNorm = (
            nn.LayerNorm(hidden_dim, eps=eps)
            if norm_type == "layer_norm"
            else ScaleNorm(eps=eps))
        self.w = nn.Parameter(torch.randn(2 * max_position_embeddings - 1))
        self.a = nn.Parameter(torch.randn(1, self.s))
        self.b = nn.Parameter(torch.randn(1, self.s))
        self.act_fn = ACT2FN[hidden_act]
        self.max_position_embeddings = max_position_embeddings
        self.softmax = net_params['softmax']
        self.relu2 = net_params['relu2']
        # self.Res_LayerNorm = nn.LayerNorm(hidden_dim,eps=eps)
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.w, std=0.02)
        nn.init.normal_(self.a, std=0.02)
        nn.init.normal_(self.b, std=0.02)

    def forward(self, x, attention_mask=None, output_attentions=False, causal=False):
        seq_len = len(x)  # ([1547, 92])
        max_position_embeddings = seq_len
        shortcut, x = x, self.LayerNorm(x)
        uv = self.uv(x)  # ([1547, 496])
        u, v, base = torch.split(self.act_fn(uv), [self.e, self.e, self.s], dim=-1)
        # Generate Query (q) and Key (k) from base.
        base = torch.einsum("...r,hr->...hr", base, self.weight) + self.bias
        q, k = torch.unbind(base, dim=-2)

        # Calculate the quadratic attention.
        # qk = torch.einsum("nd,md->nm", q, k)/np.sqrt(self.s)
        qk = torch.einsum("nd,md->nm", q, k)
        if self.relu2:
            kernel = torch.square(torch.relu(qk/np.sqrt(128)))

        if self.softmax:
            kernel = qk + attention_mask
            kernel = F.softmax(kernel, dim = -1)
        x = u * torch.einsum("nm,me->ne", kernel, v)
        x = self.o(x)
        if output_attentions:
            return x + shortcut, kernel
        return x + shortcut


class GAUNET0(nn.Module):
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
        dropout = net_params['dropout']  # 0.0
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
        self.softmax = net_params['softmax']
        self.relu2 = net_params['relu2']

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
        self.GAU_layers = nn.ModuleList([GAU(self.net_params) for _ in range(self.n_layers)])

        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(out_dim, out_dim) for idx in range(self.n_layers)])
        # self.GAU_end_layers = nn.ModuleList([GAU(self.net_params) for _ in range(self.k_layers)])
        self.gcn_end_layers = nn.ModuleList(
            [EdgeGatedGraphConv(out_dim, out_dim) for idx in range(self.k_layers)])
        self.fc = nn.Sequential(nn.Linear(out_dim, fc_features), nn.SiLU(), nn.Linear(fc_features, 1))

    def forward(self, g, lg, h, e, lh, le, _, h_soap=None):
        h = self.embedding_h(h)
        lh = self.rbf(lh)

        h = self.in_feat_dropout(h)
        if self.soap:
            h_soap = self.embedding_soap(h_soap)
            h = torch.cat([h, h_soap], dim=1)

        batch_graph_node_num = g.batch_num_nodes()

        if self.softmax:
            attn_mask = _ * _.t()
            attn_mask = attn_mask*-100000000
            start = 0
            for i in batch_graph_node_num:
                end = start + i
                attn_mask[start:end, start:end] = 0
                start = end   
              
        if self.relu2:
            attn_mask = g.ndata['_'] * g.ndata['_'].t() * 0
            start = 0
            for i in batch_graph_node_num:
                end = start + i
                attn_mask[start:end, start:end] = 1
                start = end 
  
        for gau, gcn in zip(self.GAU_layers, self.gcn_layers):
            h = gau(h, attn_mask)
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


if __name__=="__main__":
    x = torch.rand(32, 512, 768)
    gau = GAU()
    score = gau(x)
    print(len(score))
