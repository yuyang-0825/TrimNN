import igraph as ig
from igraph import Graph
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from utils import generate_labels
import os
# vertices = [[5,8],[6,7],[2,9],[2,5],[0,7],[2,9]]
# number_of_graph_vertices = 10
# number_of_vertices_labels = 6


def generate_delaunay_graph(number_of_graph_vertices, number_of_vertices_labels, pattern):
    # Simulate 10 points with coordinates
    vertices = np.random.randint(0,1000,size=[number_of_graph_vertices,2])
    # print(vertices)
    tri = Delaunay(vertices)

    # visualize Delaunay
    plt.triplot(vertices[:,0], vertices[:,1], tri.simplices)
    plt.plot(vertices[:,0], vertices[:,1], 'o',label = 1)
    # plt.show()

    # print(tri.simplices)
    # print(tri.simplices.shape)

    # adjacent matrix
    edge_list=[]
    adjmatrix = np.zeros((number_of_graph_vertices,number_of_graph_vertices))
    for triangle in tri.simplices:
        adjmatrix[triangle[0],triangle[1]] = 1
        adjmatrix[triangle[0], triangle[2]] = 1
        adjmatrix[triangle[1], triangle[2]] = 1

    # generate igraph according to adj matrix
    graph = Graph.Adjacency(adjmatrix, mode='undirected')


    pattern_label = pattern.vs["label"]
    subgraph_list = graph.get_subisomorphisms_vf2(pattern)

    subgraph = subgraph_list[0]

    graph_vertices_labels = generate_labels(number_of_graph_vertices, number_of_vertices_labels)
    graph.vs["label"] = graph_vertices_labels
    graph.es["label"] = 0

    for index, node in enumerate(subgraph):
        graph.vs[node]["label"] = int(pattern_label[index])

    # # visualize igraph
    # ig.plot(graph)

    # generate vertices labels
    # graph_vertices_labels = generate_labels(number_of_graph_vertices, number_of_vertices_labels)
    # graph.vs["label"] = graph_vertices_labels
    # graph.es["label"] = 0

    # visualize igraph
    # ig.plot(graph)
    # graph.write('graphs/test.gml')
    return graph

# save_graph_dir_p = "../benchmark/graph/NL32"
# number_of_graph_vertices = 128
# number_of_vertices_labels = 32
# for i in range(100):
#     graph_id = "G_N" + str(number_of_graph_vertices)  + "_NL" + str(number_of_vertices_labels) + "_" + str(i)
#     graph = generate_delaunay_graph(number_of_graph_vertices,number_of_vertices_labels)
#     graph.write(os.path.join(save_graph_dir_p, graph_id + ".gml"))


# number_of_graph_vertices = 16
# number_of_vertices_labels = 8
#
# # pattern = Graph(n=6, edges=[[0, 1], [0, 2], [1,2],[2,3],[2,4],[3,4],[4,5]])
# pattern = ig.read("patterns/P_6order_NL8_0.gml")
# pattern_label =pattern.vs["label"]
#
# graph = generate_delaunay_graph(number_of_graph_vertices, number_of_vertices_labels)
#
# subgraph_list = graph.get_subisomorphisms_vf2(pattern)
#
# subgraph = subgraph_list[0]
#
# graph_vertices_labels = generate_labels(number_of_graph_vertices, number_of_vertices_labels)
# graph.vs["label"] = graph_vertices_labels
# graph.es["label"] = 0
#
# for index, node in enumerate(subgraph):
#     graph.vs[node]["label"] = int(pattern_label[index])
#
#
#
