import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
import igraph as ig
from igraph import Graph
import os
import argparse
import random
from multiprocessing import Pool, cpu_count
from scipy.sparse import lil_matrix, coo_matrix
from vf2 import PatternChecker
from concurrent.futures import ProcessPoolExecutor
import csv
import multiprocessing
import time
import networkx as nx
import itertools
from tqdm import tqdm

def process_nested_list_column(column):


    sorted_list = [tuple(sorted(sublist)) for sublist in column]

    unique_sorted_list = list(set(sorted_list))

    result = [list(item) for item in sorted(unique_sorted_list)]
    return result


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


def enumerate_specific_size(initial_pattern, graph_path, result_path, labelnum):

    graph = ig.read(graph_path)
    pattern_list = patternlist(initial_pattern, labelnum)
    result = pd.DataFrame(columns=['motif','label', 'occurrence_number'])
    best_pattern_num = 0

    for pattern in tqdm(pattern_list, desc="Enumrating CC motifs"):

        pc = PatternChecker()
        subisomorphisms = pc.get_subisomorphisms(graph, pattern)
        subisomorphisms_unique = process_nested_list_column(subisomorphisms)
        result = result._append(
            {'motif': pattern, 'label': pattern.vs["label"], 'occurrence_number': len(subisomorphisms_unique)},
            ignore_index=True)
        if len(subisomorphisms_unique) >= best_pattern_num:
            best_pattern = pattern
            best_pattern_num = len(subisomorphisms_unique)

    result.to_csv(os.path.join(result_path, "Occurrence_number_size" + str(size) + '.csv'), index=False)
    best_pattern.write(os.path.join(result_path, "Overrepresented_size" + str(size) + '.gml'), format='gml')



def parse_args():
    parser = argparse.ArgumentParser(description='VF2_methods')
    parser.add_argument('-size', type=int, default=4,
                        help='specific size of CC motif (from 3 to 9)')
    parser.add_argument('-target', type=str, default='demo_data/demo_data.gml',
                        help='The path of input graph data')
    parser.add_argument('-outpath', type=str, default='result_vf2/',
                        help='folder path for output result')
    parser.add_argument('-celltype', type=int, default=8,
                        help='number of cell types')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    graph_path = args.target
    result_path = args.outpath
    labelnum = args.celltype
    size = args.size

    if not os.path.exists(result_path):
        os.makedirs(result_path)

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

    enumerate_specific_size(initial_pattern, graph_path, result_path, labelnum)
