from tqdm import tqdm
import os
import json
from jarvis.core.specie import cgcnn_feature_json
import pandas as pd
from pymatgen.core import Element, Structure
import numpy as np
from dscribe.descriptors import soap
from pymatgen.io import ase
import argparse
import dgl
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')


def build_config(config_path, nfeature_mode='basic', start=True):
    # 输入所有cubic.cif数据

    if os.path.exists('./config.json') and start:
        with open('./config.json', 'r') as f:
            config_ = json.load(f)
        return config_

    if nfeature_mode == 'cgcnn':
        with open(cgcnn_feature_json, 'r') as f:
            cgcnn_json = json.load(f)
        config_ = {"atomic_numbers": [], "node_vectors": []}
        for Z, hotfeat in cgcnn_json.items():
            config_["atomic_numbers"].append(int(Z))
            config_["node_vectors"].append(hotfeat)
        with open(config_path, 'w') as f:
            json.dump(config_, f)

    return config_


def cif_to_graph(args, data_path_root, id_prop_df, data_filename, config_, use_angle=False, radius=8, max_nbr=12):
    '''
    把单个cif转变为图的格式
    :param max_nbr:
    :param max_nbr:
    :param radius:
    :param use_angle:
    :param data_filename:
    :param data_path_root:
    :param cutoff: 截断半径，默认为8Å
    :param data_path: cif的文件所在路径以及它的名字
    :param config_: 告诉程序节点特征如何获取
    :return: a graph
    '''

    def get_distance(n1, n2):
        return np.linalg.norm(n1 - n2)

    def get_angle(e1, e2):
        e1 = [round(e, 5) for e in e1]
        e2 = [round(e, 5) for e in e2]
        e1 = np.array(e1)
        e2 = np.array(e2)
        if np.linalg.norm(e1) * np.linalg.norm(e2) != 0:
            cos_angle = e1.dot(e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            if cos_angle < -1:
                cos_angle = -1
            if cos_angle > 1:
                cos_angle = 1
        elif np.linalg.norm(e1) * np.linalg.norm(e2) == 0:
            cos_angle = 1
        return np.arccos(cos_angle)

    transformer = ase.AseAtomsAdaptor()
    crystal = Structure.from_file(os.path.join(data_path_root, data_filename))
    SOAP_atoms = transformer.get_atoms(crystal)
    SOAPor = soap.SOAP(float(args.soap_radius), n_max=int(args.n_max), l_max=int(args.l_max), species=['He'], periodic=True)
    SOAP_atoms.arrays['numbers'] = np.array(2 * np.ones(len(SOAP_atoms.arrays['numbers']), dtype=int))
    atoms = crystal.atomic_numbers
    material_id = data_filename[:-4]
    atomnum = config_['atomic_numbers']
    z_dict = {z: i for i, z in enumerate(atomnum)}
    node_feature = np.array(config_["node_vectors"])
    atom_fea = [node_feature[z_dict[atoms[i]]] for i in range(len(crystal))]
    #print(atom_fea)
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]  # 由近到远排序
    edge_index1, edge_index2, edge_attr, edge_vec = [], [], [], []
    SOAP_feature = []

    for i, nbr in enumerate(all_nbrs):
        if len(nbr) < int(max_nbr):
            SOAP_feature.append(SOAPor.create(SOAP_atoms, [i]))
            edge_index1.extend([i] * len(nbr))
            edge_index2.extend(list(map(lambda x: x[2].tolist(), nbr)))
            for ii in nbr:
                edge_attr.append(get_distance(crystal[i].coords, ii.coords))
                edge_vec.append((np.array(ii.coords) - np.array(crystal[i].coords).tolist()))
        else:
            SOAP_feature.append(SOAPor.create(SOAP_atoms, [i]))
            edge_index1.extend([i] * int(max_nbr))
            edge_index2.extend(list(map(lambda x: x[2].tolist(),
                                        nbr[:int(max_nbr)])))
            for ii in nbr[:int(max_nbr)]:
                edge_attr.append(get_distance(crystal[i].coords, ii.coords))
                edge_vec.append((np.array(ii.coords) - np.array(crystal[i].coords).tolist()))

    skip_struct=False
    for fea in atom_fea:
        if True in pd.isna(fea) or None in fea:
            skip_struct=True

    struct_info = {}
    struct_info['ids'] = material_id
    struct_info['node_feature'] = [fea.tolist() for fea in atom_fea]
    struct_info['edge_index1'] = edge_index1
    struct_info['edge_index2'] = edge_index2
    struct_info['edge_attr'] = edge_attr
  # struct_info['edge_vec'] = [vec.tolist() for vec in edge_vec]
    struct_info['SOAP_feature'] = [fea[0].tolist() for fea in SOAP_feature]

    if use_angle:
        t_index1, t_index2, t_attr = [], [], []
        for t1, i in enumerate(edge_vec):
            for t2, j in enumerate(edge_vec):
                jj = edge_index2[t1]
                jj_ = edge_index1[t2]
                if jj == jj_:
                    t_index1.append(t1)
                    t_index2.append(t2)
                    t_attr.append(get_angle(-i, j))
        struct_info['t_index1'] = t_index1
        struct_info['t_index2'] = t_index2
        struct_info['t_attr'] = t_attr

    return struct_info, skip_struct


def task_func_dgl(args):
    
    struct_graph = args[0]
    node_features = torch.tensor(struct_graph['node_feature'],dtype=torch.float32)
    # print(node_features)
    edge_list, l_edge_list = [], []
    edge_list.append(struct_graph['edge_index1'])
    edge_list.append(struct_graph['edge_index2'])
    edge_features = torch.tensor(struct_graph['edge_attr'],dtype=torch.float32)
    l_edge_list.append(struct_graph['t_index1'])
    l_edge_list.append(struct_graph['t_index2'])
    theta_features = torch.tensor(struct_graph['t_attr'],dtype=torch.float32)

    # Create the DGL Graph
    g = dgl.DGLGraph()
    g.add_nodes(len(struct_graph['node_feature']))
    g.ndata['feat'] = node_features
    g.ndata['_'] = torch.ones(len(node_features),1)
    g.ndata['soap_enc'] = torch.tensor(struct_graph['SOAP_feature'],dtype=torch.float32)
    for src, dst in np.array(edge_list).T:
        g.add_edges(src.item(), dst.item())
    g.edata['feat'] = edge_features

    lg = dgl.DGLGraph()
    lg.add_nodes(len(struct_graph['edge_attr']))
    lg.ndata['feat'] = edge_features
    for src, dst in np.array(l_edge_list).T:
        lg.add_edges(src.item(), dst.item())
    lg.edata['feat'] = theta_features
    g = (g, lg)
    label = torch.tensor(float(struct_graph['energy']))

    return g, label, struct_graph['ids']


def task_func(args):
    label = args[0]
    id_prop_df = args[1]
    data_path_root = args[2]
    Cif = args[3]
    config = args[4]
    use_angle = args[5]
    radius = args[6]
    max_nbr = args[7]
    default_args = args[8]
    struct_info, skip_struct = cif_to_graph(default_args, data_path_root,id_prop_df,
                                            Cif, config, use_angle, radius, max_nbr)
    for i in label:
        struct_info[i] = float(id_prop_df['energy'][int(struct_info['ids'])])
    return struct_info, skip_struct


def chunks(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]


def make_mul_graph(args, data_path_root, save_path, config_, label, use_angle=True,
                   radius=8, max_nbr=12,limit=None, cpu_worker_num=20):
    from multiprocessing import Pool

    id_prop_df = pd.read_csv('./id_prop.csv', sep=' ')
    cif_filename_list = os.listdir(data_path_root)
    database=[]
    if limit is None:
        for cif_list in tqdm(chunks(cif_filename_list, round(len(cif_filename_list)/30))):
            process_args = []
            for Cif in cif_list:
                process_args.append([label, id_prop_df, data_path_root, Cif, config_, use_angle, radius, max_nbr,args])
            with Pool(cpu_worker_num) as p:
                output = p.map(task_func, process_args)
                for i in output:
                    if not i[1]:
                        database.append(i[0])
    else:
        for cif_list in tqdm(chunks(cif_filename_list[:int(limit)], round(float(limit)/30))):
            process_args = []
            for Cif in cif_list:
                process_args.append([label, id_prop_df, data_path_root, Cif, config_, use_angle, radius, max_nbr,args])
            with Pool(cpu_worker_num) as p:
                output = p.map(task_func, process_args)
                for i in output:
                    if not i[1]:
                        database.append(i[0])

    print('正在进行归一化\n')
    database = max_min_process(database)

    graph_lists, graph_labels, graph_ids = [], [],[]
    for struct_list in tqdm(chunks(database, round(len(database)/30))):
        process_args = []
        for struct in struct_list:
            process_args.append([struct])
        with Pool(cpu_worker_num) as p:
            output = p.map(task_func_dgl, process_args)
            for i in output:
                graph_lists.append(i[0])
                graph_labels.append(i[1])
                graph_ids.append(i[2])
        
    data_dir = './'
    graphs = {'graph_lists': graph_lists, 'graph_labels': graph_labels, 'graph_ids':graph_ids}
    f = open(save_path, 'wb')
    pickle.dump(graphs, f)
    f.close()

#    with open(os.path.join(save_path), 'w') as f:
#        json.dump(database, f)
    return database


def argparse_():
    parser = argparse.ArgumentParser(
        description="Make graph from cif"
    )
    parser.add_argument(
        "--download_database_name",
        default="megnet",
        help="database name that you want to download"
    )
    parser.add_argument(
        "--data_path_root",
        default="./new_co_cif_final",
        help="Folder with cif file"
    )
    parser.add_argument(
        "--save_path",
        default="./graph.json",
        help="output_dir with graph.json"
    )
    parser.add_argument(
        "--config_path",
        default="./config.json",
        help="dir with config.json"
    )
    parser.add_argument(
        "--nfeature_mode",
        default="cgcnn",
        help="node feature mode for config"
    )
    parser.add_argument(
        "--download_label",
        default=['gap pbe', 'e_form'],
        help="a list, the sample label that you want to train"
    )
    parser.add_argument(
        "--train_label",
        default=['energy'],
        help="a list, the sample label that you want to train"
    )
    parser.add_argument(
        "--use_angle",
        default=True,
        help="a list, the sample label that you want to train"
    )
    parser.add_argument(
        "--cutoff_radius",
        default=8,
        help="a list, the sample label that you want to train"
    )
    parser.add_argument(
        "--max_nbr",
        default=12,
        help="max number the-nearest-neighbor"
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="max number of sample"
    )
    parser.add_argument(
        "--skip_download",
        default=False,
        help="max number the-nearest-neighbor"
    )
    parser.add_argument(
        "--config_start",
        default=True,
        help="max number the-nearest-neighbor"
    )
    parser.add_argument(
        "--n_max",
        default = 4,
        help="max_n for soap descriptor"
    )
    parser.add_argument(
        "--l_max",
        default = 4,
        help="max_l for soap descriptor"
    )
    parser.add_argument(
        "--soap_radius",
        default = 5,
        help="radius for soap descriptor"
    )
    args_ = parser.parse_args()

    return args_


def max_min_process(mp_data):
    soap = []
    for i in range(len(mp_data)):
        soap.extend(mp_data[i]['SOAP_feature'])
#    max_x = []
#    min_x = []
#    for i in range(len(x[0])):
#        max_x.append(max(np.array(x)[:, i]))
#        min_x.append(min(np.array(x)[:, i]))
    max_soap = []
    min_soap = []
    for i in range(len(soap[0])):
        max_soap.append(max(np.array(soap)[:, i]))
        min_soap.append(min(np.array(soap)[:, i]))

#    for i in range(len(mp_data)):  # 0~93
#        array_node = np.array(mp_data[i]['node_feature'])
#        mp_data[i]['node_feature'] = ((array_node - np.array(min_x))/(np.array(max_x)-np.array(min_x))).tolist()

    for i in range(len(mp_data)):  # 0~93
        array_soap = np.array(mp_data[i]['SOAP_feature'])
        mp_data[i]['SOAP_feature'] = ((array_soap - np.array(min_soap))/(np.array(max_soap)-np.array(min_soap))).tolist()

    return mp_data


if __name__ == '__main__':
    args = argparse_()
    #print(args.skip_download)
    #download_dataset(name=args.download_database_name, target=args.download_label,
    #                 cif_save_path=args.data_path_root, skip=args.skip_download)
    config = build_config(args.config_path, args.nfeature_mode, start=args.config_start)
    make_mul_graph(args, args.data_path_root, args.save_path, config_=config,
                   use_angle=args.use_angle, radius=args.cutoff_radius, max_nbr=args.max_nbr,
                   label=args.train_label, limit=args.limit)
