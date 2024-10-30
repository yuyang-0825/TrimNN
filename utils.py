import itertools
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
import os
import json
import torch
import numpy as np
import torch.nn as nn
import igraph as ig
from dgl import shortest_dist

def find_pattern_labels(lst,size):
    unique_combinations = set()


    combinations = itertools.combinations(lst, size)


    for combination in combinations:
        unique_combinations.add(tuple(sorted(combination)))

    return list(unique_combinations)



def find_combinations_with_replacement(lst,size):
    combinations = list(itertools.combinations_with_replacement(lst,size))
    return combinations

def _get_subdirs(dirpath, leaf_only=True):
    subdirs = list()
    is_leaf = True
    for filename in os.listdir(dirpath):
        if os.path.isdir(os.path.join(dirpath, filename)):
            is_leaf = False
            subdirs.extend(_get_subdirs(os.path.join(dirpath, filename), leaf_only=leaf_only))
    if not leaf_only or is_leaf:
        subdirs.append(dirpath)
    return subdirs


def _read_graphs_from_dir(dirpath):
    import igraph as ig
    graphs = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".gml":
                continue
            try:
                graph = ig.read(os.path.join(dirpath, filename))
                graph.vs["label"] = [int(x) for x in graph.vs["label"]]
                graph.es["label"] = [int(x) for x in graph.es["label"]]
                # graph.es["key"] = [int(x) for x in graph.es["key"]]
                graphs[names[0]] = graph
            except BaseException as e:
                print(e)
                break
    return graphs


# def read_graphs_from_dir(dirpath, num_workers=4):
#     graphs = dict()
#     subdirs = _get_subdirs(dirpath)
#     with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
#         results = list()
#         for subdir in subdirs:
#             results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir,))))
#         pool.close()
#
#         for subdir, x in tqdm(results):
#             x = x.get()
#             graphs[os.path.basename(subdir)] = x
#     return graphs

def read_graphs_from_dir(file_path, num_workers=4):
    graphs = dict()
    graph = ig.read(file_path)
    graph.vs["label"] = [int(x) for x in graph.vs["label"]]
    graph.es["label"] = [int(x) for x in graph.es["label"]]
    names = os.path.splitext(os.path.basename(file_path))
    graphs[names[0]] = graph
    return graphs

# def read_graphs_from_dir(dirpath, num_workers=4):
#     graphs = dict()
#     # subdirs = _get_subdirs(dirpath)
#     # with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
#     results = list()
#     # results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir,))))
#     # pool.close()
#     results.append(_read_graphs_from_dir(dirpath))
#     for x in tqdm(results):
#     #     x = x.get()
#     #     graphs[os.path.basename(subdir)] = x
#         graphs[] = x
#     return graphs


# def read_patterns_from_dir(dirpath, num_workers=4):
#     patterns = dict()
#     subdirs = _get_subdirs(dirpath)
#     with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
#         results = list()
#         for subdir in subdirs:
#             results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir,))))
#         pool.close()
#
#         for subdir, x in tqdm(results):
#             x = x.get()
#             patterns.update(x)
#     return patterns

def read_patterns_from_dir(file_path, num_workers=4):
    patterns = dict()
    graph = ig.read(file_path)
    graph.vs["label"] = [int(x) for x in graph.vs["label"]]
    graph.es["label"] = [int(x) for x in graph.es["label"]]
    names = os.path.splitext(os.path.basename(file_path))
    patterns [names[0]] = graph
    return patterns



def _read_metadata_from_dir(dirpath):
    meta = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".meta":
                continue
            try:
                with open(os.path.join(dirpath, filename), "r") as f:
                    meta[names[0]] = json.load(f)
            except BaseException as e:
                print(e)
    return meta


def read_metadata_from_dir(dirpath, num_workers=4):
    meta = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_metadata_from_dir, args=(subdir,))))
        pool.close()

        for subdir, x in tqdm(results):
            x = x.get()
            meta[os.path.basename(subdir)] = x
    return meta


def load_data(graph_dir, pattern_dir, num_workers=4):
    patterns = read_patterns_from_dir(pattern_dir, num_workers=num_workers)
    graphs = read_graphs_from_dir(graph_dir, num_workers=num_workers)
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

def get_enc_len(x, base=10):
    # return math.floor(math.log(x, base)+1.0)
    l = 0
    while x:
        l += 1
        x = x // base
    return l

def int2onehot(x, len_x, base=10):
    if isinstance(x, (int, list)):
        x = np.array(x)
    x_shape = x.shape
    x = x.reshape(-1)
    one_hot = np.zeros((len_x*base, x.shape[0]), dtype=np.float32)
    x =  x % (base**len_x)
    idx = one_hot.shape[0] - base
    while np.any(x):
        x, y = x//base, x%base
        cond = y.reshape(1, -1) == np.arange(0, base, dtype=y.dtype).reshape(base, 1)
        one_hot[idx:idx+base] = np.where(cond, 1.0, 0.0)
        idx -= base
    while idx >= 0:
        one_hot[idx] = 1.0
        idx -= base
    one_hot = one_hot.transpose(1, 0).reshape(*x_shape, len_x*base)
    return one_hot

def batch_convert_array_to_array(batch_array, max_seq_len=-1):
    batch_lens = [v.shape[0] for v in batch_array]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    result = np.zeros([len(batch_array), max_seq_len] + list(batch_array[0].shape)[1:], dtype=batch_array[0].dtype)
    for i, t in enumerate(batch_array):
        len_t = batch_lens[i]
        if len_t < max_seq_len:
            result[i, :len_t] = t
        elif len_t == max_seq_len:
            result[i] = t
        else:
            result[i] = t[:max_seq_len]
    return result

def batch_convert_tensor_to_tensor(batch_tensor, max_seq_len=-1):
    batch_lens = [v.shape[0] for v in batch_tensor]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    result = torch.zeros([len(batch_tensor), max_seq_len] + list(batch_tensor[0].size())[1:], dtype=batch_tensor[0].dtype, requires_grad=False)
    for i, t in enumerate(batch_tensor):
        len_t = batch_lens[i]
        if len_t < max_seq_len:
            result[i, :len_t].data.copy_(t)
        elif len_t == max_seq_len:
            result[i].data.copy_(t)
        else:
            result[i].data.copy_(t[:max_seq_len])
    return result

def batch_convert_len_to_mask(batch_lens, max_seq_len=-1):
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    mask = torch.ones((len(batch_lens), max_seq_len), dtype=torch.uint8, device=batch_lens[0].device, requires_grad=False)
    for i, l in enumerate(batch_lens):
        mask[i, l:].fill_(0)
    return mask

def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
    bsz = graph_sizes.size(0)
    dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

    min_size, max_size = graph_sizes.min(), graph_sizes.max()
    mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

    if min_size == max_size:
        return batched_graph_feats.view(bsz, max_size, -1), mask
    else:
        graph_sizes_list = graph_sizes.view(-1).tolist()
        unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
        for i, l in enumerate(graph_sizes_list):
            if l == max_size:
                continue
            elif l > max_size:
                unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
            else:
                mask[i, l:].fill_(0)
                zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
                unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
        return torch.stack(unbatched_graph_feats, dim=0), mask

def extend_dimensions(old_layer, new_input_dim=-1, new_output_dim=-1, upper=False):
    if isinstance(old_layer, nn.Linear):
        old_output_dim, old_input_dim = old_layer.weight.size()
        if new_input_dim == -1:
            new_input_dim = old_input_dim
        if new_output_dim == -1:
            new_output_dim = old_output_dim
        assert new_input_dim >= old_input_dim and new_output_dim >= old_output_dim

        if new_input_dim != old_input_dim or new_output_dim != old_output_dim:
            use_bias = old_layer.bias is not None
            new_layer = nn.Linear(new_input_dim, new_output_dim, bias=use_bias)
            with torch.no_grad():
                nn.init.zeros_(new_layer.weight)
                if upper:
                    new_layer.weight[:old_output_dim, :old_input_dim].data.copy_(old_layer.weight)
                else:
                    new_layer.weight[-old_output_dim:, -old_input_dim:].data.copy_(old_layer.weight)
                if use_bias:
                    nn.init.zeros_(new_layer.bias)
                    if upper:
                        new_layer.bias[:old_output_dim].data.copy_(old_layer.bias)
                    else:
                        new_layer.bias[-old_output_dim:].data.copy_(old_layer.bias)
        else:
            new_layer = old_layer
    elif isinstance(old_layer, nn.LayerNorm):
        old_input_dim = old_layer.normalized_shape
        if len(old_input_dim) != 1:
            raise NotImplementedError
        old_input_dim = old_input_dim[0]
        assert new_input_dim >= old_input_dim
        if new_input_dim != old_input_dim and old_layer.elementwise_affine:
            new_layer = nn.LayerNorm(new_input_dim, elementwise_affine=True)
            with torch.no_grad():
                nn.init.ones_(new_layer.weight)
                nn.init.zeros_(new_layer.bias)
                if upper:
                    new_layer.weight[:old_input_dim].data.copy_(old_layer.weight)
                    new_layer.bias[:old_input_dim].data.copy_(old_layer.bias)
                else:
                    new_layer.weight[-old_input_dim:].data.copy_(old_layer.weight)
                    new_layer.bias[-old_input_dim:].data.copy_(old_layer.bias)
        else:
            new_layer = old_layer
    elif isinstance(old_layer, nn.LSTM):
        old_input_dim, old_output_dim = old_layer.input_size, old_layer.hidden_size
        if new_input_dim == -1:
            new_input_dim = old_input_dim
        if new_output_dim == -1:
            new_output_dim = old_output_dim
        assert new_input_dim >= old_input_dim and new_output_dim >= old_output_dim

        if new_input_dim != old_input_dim or new_output_dim != old_output_dim:
            new_layer = nn.LSTM(new_input_dim, new_output_dim,
                num_layers=old_layer.num_layers, bidirectional=old_layer.bidirectional,
                batch_first=old_layer.batch_first, bias=old_layer.bias)
            for layer_weights in new_layer._all_weights:
                for w in layer_weights:
                    with torch.no_grad():
                        if "weight" in w:
                            new_weight = getattr(new_layer, w)
                            old_weight = getattr(old_layer, w)
                            nn.init.zeros_(new_weight)
                            if upper:
                                new_weight[:old_weight.shape[0], :old_weight.shape[1]].data.copy_(old_weight)
                            else:
                                new_weight[-old_weight.shape[0]:, -old_weight.shape[1]:].data.copy_(old_weight)
                        if "bias" in w:
                            new_bias = getattr(new_layer, w)
                            old_bias = getattr(old_layer, w)
                            if new_bias is not None:
                                nn.init.zeros_(new_bias)
                                if upper:
                                    new_bias[:old_bias.shape[0]].data.copy_(old_bias)
                                else:
                                    new_bias[-old_bias.shape[0]:].data.copy_(old_bias)
    return new_layer

_act_map = {"none": lambda x: x,
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(1/5.5),
            "prelu": nn.PReLU(),
            "gelu": nn.GELU()}


def map_activation_str_to_layer(act_str):
    try:
        return _act_map[act_str]
    except:
        raise NotImplementedError("Error: %s activation fuction is not supported now." % (act_str))

def mask_seq_by_len(x, len_x):
    x_size = list(x.size())
    if x_size[1] == len_x.max():
        mask = batch_convert_len_to_mask(len_x)
        mask_size = x_size[0:2] + [1]*(len(x_size)-2)
        x = x * mask.view(*mask_size)
    return x

def gather_indices_by_lens(lens):
    result = list()
    i, j = 0, 1
    max_j = len(lens)
    indices = np.arange(0, max_j)
    while j < max_j:
        if lens[i] != lens[j]:
            result.append(indices[i:j])
            i = j
        j += 1
    if i != j:
        result.append(indices[i:j])
    return result

def retrieve_multiple_edges(graph, source=-1, target=-1):
    if source != -1:
        e = graph.incident(source, mode=ig.OUT)
        if target != -1:
            e = set(e).intersection(graph.incident(target, mode=ig.IN))
        return ig.EdgeSeq(graph, e)
    else:
        if target != -1:
            e = graph.incident(target, mode=ig.IN)
        else:
            e = list()
        return ig.EdgeSeq(graph, e)

def get_PE(dataset):
    for data in dataset:
        graph = data["graph"]
        short_dist_graph = shortest_dist(graph, root=None)
        first_node = torch.where(short_dist_graph == short_dist_graph.max())[0][0]
        graph_distance_PE = short_dist_graph[first_node]
        data["graph"].ndata["distance_PE"] = graph_distance_PE
        pattern = data["pattern"]
        short_dist_pattern = shortest_dist(pattern, root=None)
        first_node = torch.where(short_dist_pattern == short_dist_pattern.max())[0][0]
        pattern_distance_PE = short_dist_pattern[first_node]
        data["pattern"].ndata["distance_PE"] = pattern_distance_PE/ pattern_distance_PE.max()

    return dataset