import igraph as ig
from igraph import Graph
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from utils import generate_labels
from pattern_checker import PatternChecker
from itertools import product
import random


def generate_adjacent_node_triangles_pattern(pattern_vertices_labels):   # 5 nodes
    # pattern of triangles
    pattern = Graph(n=5, edges=[[0, 1], [0, 2], [1, 2], [0, 3], [0, 4], [3, 4]])
    # ig.plot(pattern)
    # pattern_vertices_labels = generate_labels(3, number_of_vertices_labels)
    pattern.vs["label"] = pattern_vertices_labels
    pattern.es["label"] = 0
    # ig.plot(pattern)
    return pattern

# generate_triangle_pattern(6)

def label_list(number_of_vertices_labels,repeat):
    numbers = list(range(number_of_vertices_labels))
    combinations = product(numbers, repeat=5)
    combinations_list = list(combinations)
    unique_combinations = list(set(tuple(sorted(combo)) for combo in combinations_list))
    random.shuffle(unique_combinations)
    if repeat > len(unique_combinations):
        repeat = len(unique_combinations)
    else:
        repeat = repeat_number
    unique_combinations = unique_combinations[0:repeat]
    return unique_combinations, repeat



# label_list = label_list(8,10)
# print(label_list)
number_of_vertices_labels_list = [32]
repeat_number = 100
for number_of_vertices_labels in number_of_vertices_labels_list:
    labels_list, repeat = label_list(number_of_vertices_labels,repeat_number)
    for i in range(repeat):
        labels = labels_list[i]
        pattern = generate_adjacent_node_triangles_pattern(labels)
        pattern_name = "P_adjnode_NL"+str(number_of_vertices_labels)+"_"+str(i)+".gml"
        pattern_path = "/mnt/pixstor/data/yykk3/scmotif/data/large_pattern_adjnode/patterns/"
        pattern.write(pattern_path + pattern_name)
