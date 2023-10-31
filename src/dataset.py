import torch
import numpy as np
import dgl
import os
import math
import pickle
import json
import copy
import torch.utils.data as data
from collections import defaultdict, Counter
from tqdm import tqdm
from utils import get_enc_len, int2onehot, \
    batch_convert_tensor_to_tensor, batch_convert_array_to_array

INF = float("inf")



##############################################
################ Sampler Part ################
##############################################
class Sampler(data.Sampler):
    _type_map = {
        int: np.int32,
        float: np.float32}

    def __init__(self, dataset, group_by, batch_size, shuffle, drop_last):
        super(Sampler, self).__init__(dataset)
        if isinstance(group_by, str):
            group_by = [group_by]
        for attr in group_by:
            setattr(self, attr, list())
        self.data_size = len(dataset.data)
        for x in dataset.data:
            for attr in group_by:
                value = x[attr]
                if isinstance(value, dgl.DGLGraph):
                    getattr(self, attr).append(value.number_of_nodes())
                elif hasattr(value, "__len__"):
                    getattr(self, attr).append(len(value))
                else:
                    getattr(self, attr).append(value)
        self.order = copy.copy(group_by)
        self.order.append("rand")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def make_array(self):
        self.rand = np.random.rand(self.data_size).astype(np.float32)
        if self.data_size == 0:
            types = [np.float32] * len(self.order)
        else:
            types = [type(getattr(self, attr)[0]) for attr in self.order]
            types = [Sampler._type_map.get(t, t) for t in types]
        dtype = list(zip(self.order, types))
        array = np.array(
            list(zip(*[getattr(self, attr) for attr in self.order])),
            dtype=dtype)
        return array

    def __iter__(self):
        array = self.make_array()
        indices = np.argsort(array, axis=0, order=self.order)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        batch_idx = 0
        while batch_idx < len(batches)-1:
            yield batches[batch_idx]
            batch_idx += 1
        if len(batches) > 0 and (len(batches[batch_idx]) == self.batch_size or not self.drop_last):
            yield batches[batch_idx]

    def __len__(self):
        if self.drop_last:
            return math.floor(self.data_size/self.batch_size)
        else:
            return math.ceil(self.data_size/self.batch_size)


##############################################
############# EdgeSeq Data Part ##############
##############################################
class EdgeSeq:
    def __init__(self, code):
        self.u = code[:,0]
        self.v = code[:,1]
        self.ul = code[:,2]
        self.el = code[:,3]
        self.vl = code[:,4]

    def __len__(self):
        if len(self.u.shape) == 2: # single code
            return self.u.shape[0]
        else: # batch code
            return self.u.shape[0] * self.u.shape[1]

    @staticmethod
    def batch(data):
        b = EdgeSeq(torch.empty((0,5), dtype=torch.long))
        b.u = batch_convert_tensor_to_tensor([x.u for x in data])
        b.v = batch_convert_tensor_to_tensor([x.v for x in data])
        b.ul = batch_convert_tensor_to_tensor([x.ul for x in data])
        b.el = batch_convert_tensor_to_tensor([x.el for x in data])
        b.vl = batch_convert_tensor_to_tensor([x.vl for x in data])
        return b
    
    def to(self, device):
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.ul = self.ul.to(device)
        self.el = self.el.to(device)
        self.vl = self.vl.to(device)


##############################################
############# EdgeSeq Data Part ##############
##############################################
class EdgeSeqDataset(data.Dataset):
    def __init__(self, data=None):
        super(EdgeSeqDataset, self).__init__()

        if data:
            self.data = EdgeSeqDataset.preprocess_batch(data, use_tqdm=True)
        else:
            self.data = list()
        self._to_tensor()
    
    def _to_tensor(self):
        for x in self.data:
            for k in ["pattern", "graph", "subisomorphisms"]:
                if isinstance(x[k], np.ndarray):
                    x[k] = torch.from_numpy(x[k])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, filename):
        cache = defaultdict(list)
        for x in self.data:
            for k in list(x.keys()):
                if k.startswith("_"):
                    cache[k].append(x.pop(k))
        with open(filename, "wb") as f:
            torch.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        if len(cache) > 0:
            keys = cache.keys()
            for i in range(len(self.data)):
                for k in keys:
                    self.data[i][k] = cache[k][i]

    def load(self, filename):
        with open(filename, "rb") as f:
            data = torch.load(f)
        del self.data
        self.data = data

        return self

    @staticmethod
    def graph2edgeseq(graph):
        labels = graph.vs["label"]
        graph_code = list()

        for edge in graph.es:
            v, u = edge.tuple
            graph_code.append((v, u, labels[v], edge["label"], labels[u]))
        graph_code = np.array(graph_code, dtype=np.int64)
        graph_code.view(
            [("v", "int64"), ("u", "int64"), ("vl", "int64"), ("el", "int64"), ("ul", "int64")]).sort(
                axis=0, order=["v", "u", "el"])
        return graph_code

    @staticmethod
    def preprocess(x):
        pattern_code = EdgeSeqDataset.graph2edgeseq(x["pattern"])
        graph_code = EdgeSeqDataset.graph2edgeseq(x["graph"])
        subisomorphisms = np.array(x["subisomorphisms"], dtype=np.int32).reshape(-1, x["pattern"].vcount())

        x = {
            "id": x["id"],
            "pattern": pattern_code,
            "graph": graph_code,
            "counts": x["counts"],
            "subisomorphisms": subisomorphisms}
        return x
    
    @staticmethod
    def preprocess_batch(data, use_tqdm=False):
        d = list()
        if use_tqdm:
            data = tqdm(data)
        for x in data:
            d.append(EdgeSeqDataset.preprocess(x))
        return d

    @staticmethod
    def batchify(batch):
        _id = [x["id"] for x in batch]
        pattern = EdgeSeq.batch([EdgeSeq(x["pattern"]) for x in batch])
        pattern_len = torch.tensor([x["pattern"].shape[0] for x in batch], dtype=torch.int32).view(-1, 1)
        graph = EdgeSeq.batch([EdgeSeq(x["graph"]) for x in batch])
        graph_len = torch.tensor([x["graph"].shape[0] for x in batch], dtype=torch.int32).view(-1, 1)
        counts = torch.tensor([x["counts"] for x in batch], dtype=torch.float32).view(-1, 1)
        return _id, pattern, pattern_len, graph, graph_len, counts


##############################################
######### GraphAdj Data Part ###########
##############################################
class GraphAdjDataset(data.Dataset):
    def __init__(self, data=None, k=2):
        super(GraphAdjDataset, self).__init__()

        if data:
            self.data = GraphAdjDataset.preprocess_batch(data, k, use_tqdm=True)
        else:
            self.data = list()
        self._to_tensor()
    
    def _to_tensor(self):
        for x in self.data:
            for k in ["pattern", "graph"]:
                y = x[k]
                for k, v in y.ndata.items():
                    if isinstance(v, np.ndarray):
                        y.ndata[k] = torch.from_numpy(v)
                for k, v in y.edata.items():
                    if isinstance(v, np.ndarray):
                        y.edata[k] = torch.from_numpy(v)
            # if isinstance(x["subisomorphisms"], np.ndarray):
            #     x["subisomorphisms"] = torch.from_numpy(x["subisomorphisms"])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, filename):
        cache = defaultdict(list)
        for x in self.data:
            for k in list(x.keys()):
                if k.startswith("_"):
                    cache[k].append(x.pop(k))
        with open(filename, "wb") as f:
            torch.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        if len(cache) > 0:
            keys = cache.keys()
            for i in range(len(self.data)):
                for k in keys:
                    self.data[i][k] = cache[k][i]

    def load(self, filename):
        with open(filename, "rb") as f:
            data = torch.load(f)

        del self.data
        self.data = data

        return self

    @staticmethod
    def comp_indeg_norm(graph):
        import igraph as ig
        if isinstance(graph, ig.Graph):
            # 10x faster  
            in_deg = np.array(graph.indegree(), dtype=np.float32)
        elif isinstance(graph, dgl.DGLGraph):
            in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().numpy()
        else:
            raise NotImplementedError
        norm = 1.0 / in_deg
        norm[np.isinf(norm)] = 0
        return norm

    @staticmethod
    def graph2dglgraph(graph):
        # dgl.graph()
        dglgraph = dgl.DGLGraph(multigraph=True)
        dglgraph.add_nodes(graph.vcount())
        edges = graph.get_edgelist()
        dglgraph.add_edges([e[0] for e in edges], [e[1] for e in edges])
        # Deprecated
        # dglgraph.readonly(True)
        return dglgraph

    @staticmethod
    def preprocess(x, k):
        pattern = x["pattern"]
        pattern_dglgraph = GraphAdjDataset.graph2dglgraph(pattern)
        pattern_dglgraph = dgl.to_bidirected(pattern_dglgraph)
        pattern_dglgraph.ndata["indeg"] = torch.from_numpy(np.array(pattern.indegree(), dtype=np.float32))
        pattern_dglgraph.ndata["label"] = torch.from_numpy(np.array(pattern.vs["label"], dtype=np.int64))
        pattern_dglgraph.ndata["id"] = torch.from_numpy(np.arange(0, pattern.vcount(), dtype=np.int64))
        pattern_dglgraph.edata["label"] = torch.from_numpy(np.array(pattern.es["label"]*2, dtype=np.int64))

        graph = x["graph"]
        graph_dglgraph = GraphAdjDataset.graph2dglgraph(graph)
        graph_dglgraph = dgl.to_bidirected(graph_dglgraph)
        graph_dglgraph.ndata["indeg"] = torch.from_numpy(np.array(graph.indegree(), dtype=np.float32))
        graph_dglgraph.ndata["label"] = torch.from_numpy(np.array(graph.vs["label"], dtype=np.int64))
        graph_dglgraph.ndata["id"] = torch.from_numpy(np.arange(0, graph.vcount(), dtype=np.int64))
        graph_dglgraph.edata["label"] = torch.from_numpy(np.array(graph.es["label"]*2, dtype=np.int64))
        # graph_dglgraph_new = dgl.to_bidirected(graph_dglgraph)

        # subisomorphisms = np.array(x["subisomorphisms"], dtype=np.int32).reshape(-1, x["pattern"].vcount())

        x = {
            "id": x["id"],
            "pattern": pattern_dglgraph,
            "graph": graph_dglgraph,
            # "counts": x["counts"],
            # "subisomorphisms": subisomorphisms
            }
        subgraph_list = list()
        for i in range(graph_dglgraph.num_nodes()):
            subgraph, _ = dgl.khop_out_subgraph(graph_dglgraph, i, k=k)
            # subgraph.ndata["id"] = torch.from_numpy(np.arange(0, subgraph.num_nodes(), dtype=np.int64))
            subgraph_dic = {
            "id": x["id"]+'_'+str(i),
            "pattern": pattern_dglgraph,
            "graph": subgraph,
            }
            subgraph_list.append(subgraph_dic)
        # return x, subgraph_list
        return subgraph_list

    @staticmethod
    def preprocess_batch(data, k, use_tqdm=False):
        d = list()
        subgraph = list()
        if use_tqdm:
            data = tqdm(data)
        for x in data:
            # d.append(GraphAdjDataset.preprocess(x))
            subgraph.extend(GraphAdjDataset.preprocess(x, k))
        return subgraph

    @staticmethod
    def batchify(batch):
        _id = [x["id"] for x in batch]
        pattern = dgl.batch([x["pattern"] for x in batch])
        pattern_len = torch.tensor([x["pattern"].number_of_nodes() for x in batch], dtype=torch.int32).view(-1, 1)
        graph = dgl.batch([x["graph"] for x in batch])
        graph_len = torch.tensor([x["graph"].number_of_nodes() for x in batch], dtype=torch.int32).view(-1, 1)
        # counts = torch.tensor([x["counts"] for x in batch], dtype=torch.float32).view(-1, 1)
        # return _id, pattern, pattern_len, graph, graph_len, counts
        return _id, pattern, pattern_len, graph, graph_len
            
