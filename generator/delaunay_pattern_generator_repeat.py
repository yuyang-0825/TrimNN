import igraph as ig
from igraph import Graph
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from utils import generate_labels,generate_labels_new
from pattern_checker import PatternChecker

# number_of_pattern = 5
#
# number_of_vertices_labels = 3


def generate_triangle_pattern(pattern_vertices_labels):   # 3 nodes
    # pattern of triangles
    pattern = Graph(n=3,edges = [[0,1],[0,2],[1,2]])
    # ig.plot(pattern)
    # pattern_vertices_labels = generate_labels(3, number_of_vertices_labels)
    pattern.vs["label"] = pattern_vertices_labels
    pattern.es["label"] = 0
    # ig.plot(pattern)
    return pattern
    # pattern.write('patterns/test.gml')

# generate_triangle_pattern(6)

def generate_hinge_pattern(pattern_vertices_labels):  # 3 nodes
    pattern = Graph(n=3, edges=[[0, 1], [0, 2]])
    # pattern_vertices_labels = generate_labels(3, number_of_vertices_labels)
    pattern.vs["label"] = pattern_vertices_labels
    pattern.es["label"] = 0
    # ig.plot(pattern)
    return pattern

# generate_hinge_pattern(6)

def generate_adjacent_edge_triangles_pattern(pattern_vertices_labels):    # 4 nodes
    pattern = Graph(n=4, edges=[[0, 1], [0, 2], [1,2],[1,3],[2,3]])
    # pattern_vertices_labels = generate_labels(4, number_of_vertices_labels)
    pattern.vs["label"] = pattern_vertices_labels
    pattern.es["label"] = 0
    # ig.plot(pattern)
    return pattern
# generate_adjacent_edge_triangles_pattern(6)


def generate_adjacent_node_triangles_pattern(pattern_vertices_labels):    # 5 nodes
    pattern = Graph(n=5, edges=[[0, 1], [0, 2], [1,2], [0,3], [0,4], [3,4]])
    # pattern_vertices_labels = generate_labels(5, number_of_vertices_labels)
    pattern.vs["label"] = pattern_vertices_labels
    pattern.es["label"] = 0
    # ig.plot(pattern)
    return pattern
# generate_adjacent_node_triangles_pattern(6)

def generate_triangle_and_edge_pattern(pattern_vertices_labels):    # 4 nodes
    pattern = Graph(n=4, edges=[[0, 1], [0, 2], [1, 2], [0, 3]])
    # pattern_vertices_labels = generate_labels(4, number_of_vertices_labels)
    pattern.vs["label"] = pattern_vertices_labels
    pattern.es["label"] = 0
    # ig.plot(pattern)
    return pattern

pattern_path ="../data/balance_repeat/patterns/"
pattern_name = "P_triedge_NL32_5.gml"
label = [17,23,23,23]
pattern = generate_triangle_and_edge_pattern(label)
pattern.write(pattern_path+pattern_name)

