import torch.utils.data
import time
import dgl
import torch
import numpy as np
import pickle

class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_name, seed, data):
        self.data_dir = data_dir
        self.data_name = data_name
        self.seed = seed
        self.data = data

        data_idx = [idx for idx in range(len(self.data['graph_lists']))]
        self.graph_lists = [self.data['graph_lists'][i] for i in data_idx]
        self.graph_labels = [self.data['graph_labels'][i] for i in data_idx]
        self.n_samples = len(self.graph_lists)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]

class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='CO', seed=0, pklname=None):
        t0 = time.time()
        self.name = name
        self.seed = seed
        data_dir = './data/molecules/dataset'

        with open(data_dir + '/' + pklname, 'rb') as f:
            self.pkldata = pickle.load(f)

        self.num_atom_type = len(self.data['graph_lists'][0][0].ndata['feat'][0])
        self.num_bond_type = 1  # len(self.data[0]['edge_attr'][0])
        self.num_theta_type = 1
        self.soap_dim = len(self.data['graph_lists'][0][0].ndata['soap_enc'][0])
        self.data = MoleculeDGL(data_dir, self.name, self.seed,  data=self.pkldata)

        print("Time taken: {:.4f}s".format(time.time() - t0))

    def collate(self, samples):
        # The input samples is a list of pairs (graph, label). graph = [(g,lg),(g,lg),...,(g,lg)]
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        gs, lgs =[],[]
        for i in range(len(graphs)):
            g, lg = graphs[i]
            gs.append(g)
            lgs.append(lg)
        batched_graph = dgl.batch(gs)
        l_batched_graph = dgl.batch(lgs)
        return batched_graph, l_batched_graph, labels
