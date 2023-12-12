"""
    Utility file to select GraphNN model as
    selected by the user
"""
from nets.co_graph_regression.GAU import GAUNET
from nets.co_graph_regression.GAU1 import GAUNET1
from nets.co_graph_regression.GAU0 import GAUNET0
from nets.co_graph_regression.ALIGNN import ALIGNNNET
from nets.co_graph_regression.CGCNN import CGCNNNET
from nets.co_graph_regression.GATGNN import GATNet
from nets.co_graph_regression.GTF import GraphTransformerNet
import dgl
import numpy as np
import torch

def GAU(net_params):
    return GAUNET(net_params)
def ALIGNN(net_params):
    return ALIGNNNET(net_params)
def CGCNN(net_params):
    return CGCNNNET(net_params)
def GAU0(net_params): # 没有lgcn
    return GAUNET0(net_params)
def GAU1(net_params): # 没有lgcn 多头
    return GAUNET1(net_params)
def GATGNN(net_params):
    return GATNet(net_params)
def GTF(net_params):
    return GraphTransformerNet(net_params)
def MEGNet(net_params):
    return MEGNetModel(nfeat = net_params['num_atom_type'], ntarget=2, graph_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5))
def SchNet1(net_params):
    return SchNet(n_interactions=3, n_atom_basis=128, n_filters=128, n_gaussians=25, cutoff=5.0)
def OrbNet1(net_params):
    return OrbNet(n_orbital=64, n_hidden=128, n_output=1)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GAU': GAU,
        'GAU0': GAU0,
        'GAU1': GAU1,
        'ALIGNN': ALIGNN,
        'CGCNN': CGCNN,
        'GATGNN': GATGNN,
        'GTF': GTF,
        'MEGNet': MEGNet,
        'SchNet': SchNet1,
        'OrbNet': OrbNet1
    }
        
    return models[MODEL_NAME](net_params)