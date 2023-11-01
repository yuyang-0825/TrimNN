import torch
import os
import numpy as np
import dgl
import logging
import datetime
import math
import sys
import gc
import json
import igraph as ig
from igraph import Graph
import time
import csv
import torch.nn.functional as F
import warnings
import itertools
from collections import OrderedDict
from torch.utils.data import DataLoader
import argparse
from rgin import RGIN
from dataset import Sampler, EdgeSeqDataset, GraphAdjDataset
warnings.filterwarnings("ignore")


config = {
    "max_npv": 8,  # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 8,  # max_number_pattern_edges: 8, 16, 32
    "max_npvl": 32,  # max_number_pattern_vertex_labels: 8, 16, 32
    "max_npel": 8,  # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 128,  # max_number_graph_vertices: 64, 512,4096
    "max_nge": 1024,  # max_number_graph_edges: 256, 2048, 16384
    "max_ngvl": 32,  # max_number_graph_vertex_labels: 16, 64, 256
    "max_ngel": 16,  # max_number_graph_edge_labels: 16, 64, 256

    "base": 2,

    "gpu_id": 0,
    "num_workers": 12,

    "epochs": 100,
    "batch_size": 128,
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

    "model": "RGIN",  # CNN, RNN, TXL, RGCN, RGIN, RSIN

    "emb_dim": 128,
    "activation_function": "leaky_relu",  # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu

    "filter_net": "MaxGatedFilterNet",  # None, MaxGatedFilterNet
    "predict_net": "DIAMNet",  # MeanPredictNet, SumPredictNet, MaxPredictNet,
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
            # x["subisomorphisms"] = meta[p][g]["subisomorphisms"]
            # x["counts"] = meta[p][g]["counts"]

            # g_idx = int(g.rsplit("_", 1)[-1])
            # if g_idx % 10 == 0:
            #     dev_data.append(x)
            # elif g_idx % 10 == 1:
            #     test_data.append(x)
            # else:
            #     train_data.append(x)
            test_data.append(x)

    return OrderedDict({"test": test_data})


def enumerate_triangle(graph_path,model_path,result_path,labelnum,k):
    initial_pattern = Graph(n=3, edges=[[0, 1], [0, 2], [1, 2]])
    celltype = list(range(0, labelnum))
    combinations = list(itertools.combinations_with_replacement(celltype, 3))
    for labels in combinations:
        initial_pattern.vs["label"] = labels
        initial_pattern.es["label"] = 0
        initial_pattern["type"] = labels

        data = load_data(graph_path, initial_pattern)
        data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
        for data_type, x in data.items():
            dataset = GraphAdjDataset(x, k)

        sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=config["batch_size"],
                          shuffle=data_type == "train", drop_last=False)
        data_loader = DataLoader(dataset,
                                 batch_sampler=sampler,
                                 collate_fn=GraphAdjDataset.batchify,
                                 pin_memory=data_type == "train")

        data_loaders[data_type] = data_loader
        device = torch.device("cuda:%d" % config["gpu_id"] if config["gpu_id"] != -1 else "cpu")
        model = RGIN(config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        evaluate_results = evaluate(model, data_type, data_loader, device, config)
        pred_exist = evaluate_results["data"]["pred_exist"]

        with open(result_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([labels, pred_exist])


def enumerate_4order_one(graph_path, model_path, result_path, labelnum,k):
    initial_pattern = Graph(n=4, edges=[[0, 1], [0, 2], [1, 2], [0, 3]])
    celltype = list(range(0, labelnum))
    combinations = list(itertools.product(celltype, repeat=4))
    for labels in combinations:
        initial_pattern.vs["label"] = labels
        initial_pattern.es["label"] = 0
        initial_pattern["type"] = labels

        data = load_data(graph_path, initial_pattern)
        data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
        for data_type, x in data.items():
            dataset = GraphAdjDataset(x, k)

        sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=config["batch_size"],
                          shuffle=data_type == "train", drop_last=False)

        data_loader = DataLoader(dataset,
                                 batch_sampler=sampler,
                                 collate_fn=GraphAdjDataset.batchify,
                                 pin_memory=data_type == "train")

        data_loaders[data_type] = data_loader
        device = torch.device("cuda:%d" % config["gpu_id"] if config["gpu_id"] != -1 else "cpu")
        model = RGIN(config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        evaluate_results = evaluate(model, data_type, data_loader, device, config)
        pred_exist = evaluate_results["data"]["pred_exist"]

        with open(result_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([labels, pred_exist])


def enumerate_4order_two(graph_path, model_path, result_path, labelnum,k):
    initial_pattern = Graph(n=4, edges=[[0, 1], [0, 2], [1, 2], [1, 3],[2, 3]])
    celltype = list(range(0, labelnum))
    combinations = list(itertools.product(celltype, repeat=4))
    for labels in combinations:
        initial_pattern.vs["label"] = labels
        initial_pattern.es["label"] = 0
        initial_pattern["type"] = labels

        data = load_data(graph_path, initial_pattern)
        data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
        for data_type, x in data.items():
            dataset = GraphAdjDataset(x, k)

        sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=config["batch_size"],
                          shuffle=data_type == "train", drop_last=False)
        data_loader = DataLoader(dataset,
                                 batch_sampler=sampler,
                                 collate_fn=GraphAdjDataset.batchify,
                                 pin_memory=data_type == "train")

        data_loaders[data_type] = data_loader
        device = torch.device("cuda:%d" % config["gpu_id"] if config["gpu_id"] != -1 else "cpu")
        model = RGIN(config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        evaluate_results = evaluate(model, data_type, data_loader, device, config)
        pred_exist = evaluate_results["data"]["pred_exist"]

        with open(result_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([labels, pred_exist])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, help='k-hop number')
    parser.add_argument('-graph', type=str, help='triangulated graph path')
    parser.add_argument('-nodetype', type=int, help='number of node types')
    parser.add_argument('-outpath', type=str, help='output path')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # k-hop
    args = parse_args()
    k = args.k
    graph_dir = args.graph
    model_path = 'model/epoch97_triangle+triedge.pt'

    # celltype number
    labelnum = args.nodetype

    # result_path
    result_path = args.outpath


    enumerate_triangle(graph_dir, model_path, result_path, labelnum, k)
    # enumerate_4order_one(gir_dir, model_path, result_4order_one, labelnum, k)
    # enumerate_4order_two(graph_dir, model_path, result_4order_two, labelnum, k)

