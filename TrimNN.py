import torch
import os
import numpy as np
import dgl
import logging
import datetime
import math
import sys
import gc
import networkx as nx
import igraph as ig
from igraph import Graph
import time
import csv
import warnings
import pandas as pd
import itertools
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from rgin import RGIN
from utils import get_PE
from dataset import Sampler, EdgeSeqDataset, GraphAdjDataset
warnings.filterwarnings("ignore")
import argparse


config = {
    "max_npv": 10,  # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 8,  # max_number_pattern_edges: 8, 16, 32
    "max_npvl": 32,  # max_number_pattern_vertex_labels: 8, 16, 32
    "max_npel": 1,  # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 128,  # max_number_graph_vertices: 64, 512,4096
    "max_nge": 1024,  # max_number_graph_edges: 256, 2048, 16384
    "max_ngvl": 32,  # max_number_graph_vertex_labels: 16, 64, 256
    "max_ngel": 1,  # max_number_graph_edge_labels: 16, 64, 256

    "base": 2,

    "gpu_id": -1,
    "num_workers": 12,

    "epochs": 100,
    "batch_size": 1024,
    "update_every": 1,  # actual batch_sizer = batch_size * update_every
    "print_every": 100,
    "init_emb": "Equivariant",  # None, Orthogonal, Normal, Equivariant
    "share_emb": True,  # sharing embedding requires the same vector length
    "share_arch": True,  # sharing architectures
    "dropout": 0.2,
    "dropatt": 0.2,

    "reg_loss": "MSE",  # MAE, MSEl
    "bp_loss": "MSE",  # MAE, MSE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",  # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01,
    # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
    # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.001,
    "weight_decay": 0.00001,
    "max_grad_norm": 8,

    "model": "Transform",  # CNN, RNN, TXL, RGCN, RGIN, RSIN

    "emb_dim": 128,
    "activation_function": "leaky_relu",  # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu

    "filter_net": "MaxGatedFilterNet",  # None, MaxGatedFilterNet
    "predict_net": "MaxPredictNet",  # MeanPredictNet, SumPredictNet, MaxPredictNet,
    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
    # DIAMNet
    "predict_net_add_enc": True,
    "predict_net_add_degree": True,
    "predict_net_hidden_dim": 128,
    "predict_net_num_heads": 4,
    "predict_net_mem_len": 4,
    "predict_net_mem_init": "mean",
    # mean, sum, max, attn, circular_mean, circular_sum, circular_max, circular_attn, lstm
    "predict_net_recurrent_steps": 3,

    "cnn_hidden_dim": 128,
    "cnn_conv_channels": (128, 128, 128),
    "cnn_conv_kernel_sizes": (2, 3, 4),
    "cnn_conv_strides": (1, 1, 1),
    "cnn_conv_paddings": (1, 1, 1),
    "cnn_pool_kernel_sizes": (2, 3, 4),
    "cnn_pool_strides": (1, 1, 1),
    "cnn_pool_paddings": (0, 1, 2),

    "rnn_type": "LSTM",  # GRU, LSTM
    "rnn_bidirectional": False,
    "rnn_graph_num_layers": 3,
    "rnn_pattern_num_layers": 3,
    "rnn_hidden_dim": 128,

    "txl_graph_num_layers": 3,
    "txl_pattern_num_layers": 3,
    "txl_d_model": 128,
    "txl_d_inner": 128,
    "txl_n_head": 4,
    "txl_d_head": 4,
    "txl_pre_lnorm": True,
    "txl_tgt_len": 64,
    "txl_ext_len": 0,  # useless in current settings
    "txl_mem_len": 64,
    "txl_clamp_len": -1,  # max positional embedding index
    "txl_attn_type": 0,  # 0 for Dai et al, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
    "txl_same_len": False,

    "rgcn_num_bases": 8,
    "rgcn_regularizer": "bdd",  # basis, bdd
    "rgcn_graph_num_layers": 3,
    "rgcn_pattern_num_layers": 3,
    "rgcn_hidden_dim": 128,
    "rgcn_ignore_norm": False,  # ignorm=True -> RGCN-SUM

    "rgin_num_bases": 8,
    "rgin_regularizer": "bdd",  # basis, bdd
    "rgin_graph_num_layers": 3,
    "rgin_pattern_num_layers": 3,
    "rgin_hidden_dim": 128,
    "distance_PE": True,
    "PE_alpha": 0.5,

    "gt_num_layers": 4,
    "gt_hidden_dim": 80,
    "gt_num_heads": 8,
    "pos_enc": False,
    "pos_enc_size": 2,
}
def evaluate(model, data_type, data_loader, device, config, logger=None, writer=None):
    epoch_step = len(data_loader)
    # total_step = config["epochs"] * epoch_step
    total_cnt = 1e-6

    evaluate_results = {"data": {"id": list(), "pred": list(), "pred_exist": 0},
                        # "time": {"avg": list(), "total": 0.0},
                        "time": { "total": 0.0}
                        }
    model.eval()

    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            ids, pattern, pattern_len, graph, graph_len = batch
            cnt = graph_len.shape[0]
            total_cnt += cnt

            evaluate_results["data"]["id"].extend(ids)
            # evaluate_results["data"]["counts"].extend(counts.view(-1).tolist())

            pattern = pattern.to(device)
            graph = graph.to(device)
            pattern_len, graph_len = pattern_len.to(device), graph_len.to(device)

            st = time.time()
            pred = model(pattern, pattern_len, graph, graph_len)
            pred_exist = torch.where(pred > 0, 1.0, 0.0)
            et = time.time()
            evaluate_results["time"]["total"] += (et - st)
            avg_t = (et - st) / (cnt + 1e-8)
            # evaluate_results["time"]["avg"].extend([avg_t] * cnt)
            evaluate_results["data"]["pred"].extend(pred.cpu().view(-1).tolist())
            evaluate_results["data"]["pred_exist"] += torch.sum(pred_exist).cpu().tolist()
    gc.collect()
    return evaluate_results


def read_patterns(pattern):
    patterns = dict()
    graph = pattern
    graph.vs["label"] = [int(x) for x in graph.vs["label"]]
    graph.es["label"] = [int(x) for x in graph.es["label"]]
    # names = os.path.splitext(os.path.basename(file_path))
    patterns['enumerate'] = graph
    return patterns

def read_graphs_from_dir(file_path):
    graphs = dict()
    graph = ig.read(file_path)
    graph.vs["label"] = [int(x) for x in graph.vs["label"]]
    graph.es["label"] = [int(x) for x in graph.es["label"]]
    names = os.path.splitext(os.path.basename(file_path))
    graphs[names[0]] = graph
    return graphs

def load_data(graph_dir, pattern):
    patterns = read_patterns(pattern)
    graphs = read_graphs_from_dir(graph_dir)
    # meta = read_metadata_from_dir(metadata_dir, num_workers=num_workers)

    # train_data, dev_data, test_data = list(), list(), list()
    test_data = list()
    for p, pattern in patterns.items():
        for g, graph in graphs.items():
            x = dict()
            x["id"] = ("%s-%s" % (p, g))
            x["pattern"] = pattern
            x["graph"] = graph
            test_data.append(x)

    return OrderedDict({"test": test_data})


def subgraph_matching(pattern, graph_path, model_path, result_path, k):
    # graph_names = os.listdir(graph_path)

    data = load_data(graph_path, pattern)
    data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
    for data_type, x in data.items():
        dataset = GraphAdjDataset(x, k)
        dataset = get_PE(dataset)

    sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=config["batch_size"],
                      shuffle=data_type == "train", drop_last=False)
    data_loader = DataLoader(dataset,
                             batch_sampler=sampler,
                             collate_fn=GraphAdjDataset.batchify,
                             pin_memory=data_type == "train")

    data_loaders[data_type] = data_loader
    # device = torch.device("cuda:%d" % config["gpu_id"] if config["gpu_id"] != -1 else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGIN(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    evaluate_results = evaluate(model, data_type, data_loader, device, config)
    pred_exist = evaluate_results["data"]["pred_exist"]

    with open(result_path+'predicted_result.txt', 'w') as file:
        file.write(f"Predicted occurrence number: {pred_exist}\n")

def patternlist(initial_pattern,labelnum):
    celltype = list(range(0, labelnum))
    combinations = list(itertools.product(celltype, repeat=len(initial_pattern.vs)))

    pattern_list = []

    for labels in combinations:
        pattern = initial_pattern.copy()
        pattern.vs["label"] = labels
        pattern.es["label"] = 0
        pattern["type"] = labels

        edges = pattern.get_edgelist()
        g_networkx = nx.Graph()
        g_networkx.add_edges_from(edges)
        if nx.is_planar(g_networkx):
            isomorphic = 0
            for i, graph in enumerate(pattern_list):
                if pattern.isomorphic_vf2(graph, color1=pattern.vs["label"], color2=graph.vs["label"]):
                    isomorphic += 1
            if isomorphic == 0:
                pattern_list.append(pattern)
    return pattern_list

def enumerate_specific_size(initial_pattern, graph_path, model_path, result_path, labelnum, k,size):
    # graph_names = os.listdir(graph_path)

    pattern_list = patternlist(initial_pattern, labelnum)
    # for labels in left_tuple:
    best_pattern_num = 0
    best_pattern = initial_pattern.copy()
    result = pd.DataFrame(columns=['motif', 'predicted_occurrence_number'])
    for pattern in tqdm(pattern_list, desc="Enumrating CC motifs"):

        pattern_pred = 0


        data = load_data(graph_path, pattern)
        data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
        for data_type, x in data.items():
            dataset = GraphAdjDataset(x, k)
            dataset = get_PE(dataset)

        sampler = Sampler(dataset, group_by=["graph",'label', "pattern"], batch_size=config["batch_size"],
                          shuffle=data_type == "train", drop_last=False)
        data_loader = DataLoader(dataset,
                                 batch_sampler=sampler,
                                 collate_fn=GraphAdjDataset.batchify,
                                 pin_memory=data_type == "train")

        data_loaders[data_type] = data_loader
        # device = torch.device("cuda:%d" % config["gpu_id"] if config["gpu_id"] != -1 else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RGIN(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        evaluate_results = evaluate(model, data_type, data_loader, device, config)
        pred_exist = evaluate_results["data"]["pred_exist"]

        pattern_pred += pred_exist
        result = result._append({'motif': pattern,'label':pattern.vs["label"], 'predicted_occurrence_number': pattern_pred},
                                ignore_index=True)

        if pattern_pred >= best_pattern_num:
            best_pattern = pattern
            best_pattern_num = pattern_pred

    result.to_csv(os.path.join(result_path,"Predicted_occurrence_size"+str(size)+'.csv'), index=False)
    best_pattern.write(os.path.join(result_path,"Overrepresented_size"+str(size)+'.gml'), format='gml')


def enumerate_triangle(graph_path,model_path,labelnum,k):
    initial_pattern = Graph(n=3, edges=[[0, 1], [0, 2], [1, 2]])
    celltype = list(range(0, labelnum))
    combinations = list(itertools.combinations_with_replacement(celltype, 3))

    # graph_names = os.listdir(graph_path)

    max_occur = 0
    max_pattern = initial_pattern.copy()
    for labels in tqdm(combinations, desc="Enumrating CC motifs"):
        initial_pattern.vs["label"] = labels
        initial_pattern.es["label"] = 0
        initial_pattern["type"] = labels
        pattern_pred = 0

        # for graph_name in graph_names:
        data = load_data(graph_path, initial_pattern)
        data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
        for data_type, x in data.items():
            dataset = GraphAdjDataset(x, k)
            dataset = get_PE(dataset)

        sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=config["batch_size"],
                          shuffle=data_type == "train", drop_last=False)
        data_loader = DataLoader(dataset,
                                 batch_sampler=sampler,
                                 collate_fn=GraphAdjDataset.batchify,
                                 pin_memory=data_type == "train")

        data_loaders[data_type] = data_loader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RGIN(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        evaluate_results = evaluate(model, data_type, data_loader, device, config)
        pred_exist = evaluate_results["data"]["pred_exist"]

        pattern_pred += pred_exist

        if pattern_pred > max_occur:
            max_pattern = initial_pattern.copy()
    return max_pattern

def generate_patterns_with_new_node(best_pattern,graph_path,labelnum):

    # node number of original graph
    num_vertices = len(best_pattern.vs)

    all_combinations = []
    # graph = ig.read(graph_path)
    for r in range(1, num_vertices + 1):
        all_combinations.extend(list(itertools.combinations(range(num_vertices), r)))

    new_graphs = []

    for label in range(labelnum):
        for combination in all_combinations:

            g = best_pattern.copy()

            g.add_vertices(1)

            # index of new node
            new_vertex_index = num_vertices
            g.vs[new_vertex_index]["label"] = label

            # add new edge
            edges_to_add = [(new_vertex_index, i) for i in combination]
            g.add_edges(edges_to_add)
            g.es["label"] = 0

            # new graph list
            # if graph.subisomorphic_vf2(g,color1 = graph.vs["label"],color2 = g.vs["label"]):
            #     new_graphs.append(g)

            edges = g.get_edgelist()
            g_networkx = nx.Graph()
            g_networkx.add_edges_from(edges)
            if nx.is_planar(g_networkx):
                isomorphic = 0
                for i, graph in enumerate(new_graphs):
                    if g.isomorphic_vf2(graph, color1=g.vs["label"], color2=graph.vs["label"]):
                        isomorphic += 1
                if isomorphic == 0:
                    new_graphs.append(g)

    return new_graphs


def enumerate_all_size(size,graph_path, model_path, labelnum,k,result_path):

    triangle_pattern = enumerate_triangle(graph_path,model_path,labelnum,k)
    triangle_pattern.write(os.path.join(result_path,"Overrepresented_size3.gml"), format='gml')

    now_size = triangle_pattern.vcount()
    best_pattern = triangle_pattern

    while(now_size<size):

        best_pattern_num = 0
        result = pd.DataFrame(columns=['graph','label', 'pattern_pred'])
        new_patternlist = generate_patterns_with_new_node(best_pattern,graph_path,labelnum)

        for new_pattern in tqdm(new_patternlist, desc="Searching CC motifs"):
            data = load_data(graph_path, new_pattern)
            data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
            for data_type, x in data.items():
                dataset = GraphAdjDataset(x, k)
                dataset = get_PE(dataset)

            sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=config["batch_size"],
                              shuffle=data_type == "train", drop_last=False)
            data_loader = DataLoader(dataset,
                                     batch_sampler=sampler,
                                     collate_fn=GraphAdjDataset.batchify,
                                     pin_memory=data_type == "train")

            data_loaders[data_type] = data_loader
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = RGIN(config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            evaluate_results = evaluate(model, data_type, data_loader, device, config)
            pattern_pred = evaluate_results["data"]["pred_exist"]
            # print(pattern_pred)
            result = result._append({'graph': new_pattern,'label':new_pattern.vs["label"], 'pattern_pred': pattern_pred}, ignore_index=True)
            if pattern_pred >= best_pattern_num:
                best_pattern = new_pattern
                best_pattern_num = pattern_pred

        now_size+=1
        best_pattern.write(os.path.join(result_path,"Overrepresented_size" + str(now_size) + ".gml"), format='gml')
    return best_pattern


def parse_args():
    parser = argparse.ArgumentParser(description='TrimNN_methods')
    parser.add_argument('-size', type=int, default=4,
                        help='specific size of CC motif (from 3 to 9)')
    parser.add_argument('-motif', type=str, default='demo_data/size-3.gml',
                        help='The path of input motif data')
    parser.add_argument('-k', type=int, default=2,
                        help='k-hop')
    parser.add_argument('-target', type=str, default='demo_data/demo_data.gml',
                        help='The path of input graph data')
    parser.add_argument('-outpath', type=str, default='result/',
                        help='folder path for output result')
    parser.add_argument('-function', type=str, default='subgraph_matching',
                        help='folder path for output result')
    parser.add_argument('-search', type=str, default='greedy',
                        help='Search method for motif growth, now is greedy')
    parser.add_argument('-celltype', type=int, default=8,
                        help='number of cell types')


    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # k-hop
    args = parse_args()
    k = args.k
    graph_path = args.target
    result_path = args.outpath
    motif_path = args.motif
    labelnum = args.celltype
    search = args.search
    function = args.function
    size = args.size

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model_path = 'model/TrimNN.pt'

    motif = ig.read(motif_path)

    if function == 'subgraph_matching':
        subgraph_matching(motif, graph_path, model_path, result_path, k)
    elif function == 'specific_size':
        if size == 3:
            initial_pattern = Graph(n=3, edges=[[0, 1], [0, 2], [1, 2]])
        elif size == 4:
            initial_pattern = Graph(n=4, edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
        elif size == 5:
            initial_pattern = Graph(n=5, edges=[[0, 1], [0, 2], [1, 2], [0, 3], [0, 4], [3, 4]])
        elif size == 6:
            initial_pattern = Graph(n=6, edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4], [4, 5], [3, 5]])
        elif size == 7:
            initial_pattern = Graph(n=7, edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [5, 6]])
        elif size == 8:
            initial_pattern = Graph(n=8, edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [5, 6],
                                                [5, 7], [6, 7]])
        elif size == 9:
            initial_pattern = Graph(n=9, edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [6, 7],
                                                [6, 8], [7, 8]])
        enumerate_specific_size(initial_pattern, graph_path, model_path, result_path, labelnum, k, size)
    elif function == 'all_size':
        if search == "greedy":
            enumerate_all_size(size, graph_path, model_path, labelnum, k, result_path)
        else:
            print("Now we only support greedy method for searching ,more methods are in development")
    else:
        print("Don't have this function")


