import igraph as ig
from igraph import Graph
import numpy as np
import os
import json
from delaunay_pattern_generator import generate_hinge_pattern, generate_triangle_pattern, generate_adjacent_edge_triangles_pattern, generate_adjacent_node_triangles_pattern, generate_triangle_and_edge_pattern
from delaunay_graph_generator import generate_delaunay_graph
from pattern_checker import PatternChecker
from multiprocessing import Pool, cpu_count

# number_of_graph_vertices_list = [16, 32, 64, 128]
# number_of_vertices_labels_list = [8, 16, 32]
# number_of_vertices_labels_list = [8]
# pattern_type_list = ["triangle", "hinge", "adjedge", "adjnode", "triedge"]
# pattern_type_list = ["triangle"]
# pattern_type_list = ["triangle"]
# number_of_graph_vertices_list = [8]
# number_of_vertices_labels_list = [2]
# pattern_type_list = ["triangle"]

# pattern_random = 6
# graph_random = 4000
# positive_counts = graph_random/2
# negative_counts = graph_random/2


# for p in range(pattern_random):
#     for pattern_type in pattern_type_list:
#         for number_of_vertices_labels in number_of_vertices_labels_list:
#             if pattern_type == "triangle":
#                 pattern = generate_triangle_pattern(number_of_vertices_labels)
#             elif pattern_type == "hinge":
#                 pattern = generate_hinge_pattern(number_of_vertices_labels)
#             elif pattern_type == "adjedge":
#                 pattern = generate_adjacent_edge_triangles_pattern(number_of_vertices_labels)
#             elif pattern_type == "adjnode":
#                 pattern = generate_adjacent_node_triangles_pattern(number_of_vertices_labels)
#             elif pattern_type == "triedge":
#                 pattern = generate_triangle_and_edge_pattern(number_of_vertices_labels)
#             else:
#                 print("error type")
#             pattern_dir ="../data/balance/patterns/"
#             if not os.path.isdir(pattern_dir):
#                 os.mkdir(pattern_dir)
#             pattern.write(pattern_dir+"P_"+pattern_type+"_NL"+str(number_of_vertices_labels)+"_"+str(p)+".gml")
#
#             # for g in range(graph_random):
#             for number_of_graph_vertices in number_of_graph_vertices_list:
#                 if number_of_graph_vertices < number_of_vertices_labels:
#                     continue
#                 positive_counts = 0
#                 negative_counts = 0
#                 g = 0
#                 while positive_counts < graph_random/2 or negative_counts < graph_random/2:
#                     graph = generate_delaunay_graph(number_of_graph_vertices, number_of_vertices_labels)
#
#                     save_graph_dir_p = "../data/balance/graphs/P_" + pattern_type+"_NL"+str(number_of_vertices_labels) +"_"+str(p)+ "/"
#                     save_metadata_dir_p = "../data/balance/metadata/P_" + pattern_type + "_NL" + str(
#                         number_of_vertices_labels) + "_" + str(p) + "/"
#                     if not os.path.isdir(save_metadata_dir_p):
#                         os.mkdir(save_metadata_dir_p)
#
#                     graph_id = "G_N"+str(number_of_graph_vertices)+"_"+pattern_type+"_NL"+str(number_of_vertices_labels)+"_"+str(g)
#                     if not os.path.isdir(save_graph_dir_p):
#                         os.mkdir(save_graph_dir_p)
#
#
#                     pc = PatternChecker()
#                     subisomorphisms = pc.get_subisomorphisms(graph, pattern)
#                     metadata = {"counts": len(subisomorphisms), "subisomorphisms": subisomorphisms}
#
#                     if len(subisomorphisms) == 0:
#                         negative_counts += 1
#                     else:
#                         positive_counts += 1
#
#                     if len(subisomorphisms) == 0 and negative_counts <= graph_random/2:
#                         g += 1
#                         graph.write(os.path.join(save_graph_dir_p, graph_id + ".gml"))
#                         with open(os.path.join(save_metadata_dir_p, graph_id + ".meta"), "w") as f:
#                             json.dump(metadata, f)
#                     elif len(subisomorphisms) != 0 and positive_counts <= graph_random/2:
#                         g += 1
#                         graph.write(os.path.join(save_graph_dir_p, graph_id + ".gml"))
#                         with open(os.path.join(save_metadata_dir_p, graph_id + ".meta"), "w") as f:
#                             json.dump(metadata, f)
#


# def generate(pattern_type, number_of_vertices_labels, p):
#
#     graph_random = 4000
#     if pattern_type == "triangle":
#         pattern = generate_triangle_pattern(number_of_vertices_labels)
#     elif pattern_type == "hinge":
#         pattern = generate_hinge_pattern(number_of_vertices_labels)
#     elif pattern_type == "adjedge":
#         pattern = generate_adjacent_edge_triangles_pattern(number_of_vertices_labels)
#     elif pattern_type == "adjnode":
#         pattern = generate_adjacent_node_triangles_pattern(number_of_vertices_labels)
#     elif pattern_type == "triedge":
#         pattern = generate_triangle_and_edge_pattern(number_of_vertices_labels)
#     else:
#         print("error type")
#     pattern_dir = "../data/balance2/patterns/"
#     if not os.path.isdir(pattern_dir):
#         os.mkdir(pattern_dir)
#     pattern.write(pattern_dir + "P_" + pattern_type + "_NL" + str(number_of_vertices_labels) + "_" + str(p) + ".gml")
#
#     # for g in range(graph_random):
#     for number_of_graph_vertices in number_of_graph_vertices_list:
#         if number_of_graph_vertices < number_of_vertices_labels:
#             continue
#         positive_counts = 0
#         negative_counts = 0
#         g = 0
#         while positive_counts < graph_random / 2 or negative_counts < graph_random / 2:
#             graph = generate_delaunay_graph(number_of_graph_vertices, number_of_vertices_labels)
#
#             save_graph_dir_p = "../data/balance2/graphs/P_" + pattern_type + "_NL" + str(
#                 number_of_vertices_labels) + "_" + str(p) + "/"
#             save_metadata_dir_p = "../data/balance2/metadata/P_" + pattern_type + "_NL" + str(
#                 number_of_vertices_labels) + "_" + str(p) + "/"
#             if not os.path.isdir(save_metadata_dir_p):
#                 os.mkdir(save_metadata_dir_p)
#
#             graph_id = "G_N" + str(number_of_graph_vertices) + "_" + pattern_type + "_NL" + str(
#                 number_of_vertices_labels) + "_" + str(g)
#             if not os.path.isdir(save_graph_dir_p):
#                 os.mkdir(save_graph_dir_p)
#
#             pc = PatternChecker()
#             subisomorphisms = pc.get_subisomorphisms(graph, pattern)
#             metadata = {"counts": len(subisomorphisms), "subisomorphisms": subisomorphisms}
#
#             if len(subisomorphisms) == 0:
#                 negative_counts += 1
#             else:
#                 positive_counts += 1
#
#             if len(subisomorphisms) == 0 and negative_counts <= graph_random / 2:
#                 g += 1
#                 graph.write(os.path.join(save_graph_dir_p, graph_id + ".gml"))
#                 with open(os.path.join(save_metadata_dir_p, graph_id + ".meta"), "w") as f:
#                     json.dump(metadata, f)
#             elif len(subisomorphisms) != 0 and positive_counts <= graph_random / 2:
#                 g += 1
#                 graph.write(os.path.join(save_graph_dir_p, graph_id + ".gml"))
#                 with open(os.path.join(save_metadata_dir_p, graph_id + ".meta"), "w") as f:
#                     json.dump(metadata, f)


def generate(pattern_type, number_of_vertices_labels,  number_of_graph_vertices, p):

    graph_random = 4000
    # if pattern_type == "triangle":
    #     pattern = generate_triangle_pattern(number_of_vertices_labels)
    # elif pattern_type == "hinge":
    #     pattern = generate_hinge_pattern(number_of_vertices_labels)
    # elif pattern_type == "adjedge":
    #     pattern = generate_adjacent_edge_triangles_pattern(number_of_vertices_labels)
    # elif pattern_type == "adjnode":
    #     pattern = generate_adjacent_node_triangles_pattern(number_of_vertices_labels)
    # elif pattern_type == "triedge":
    #     pattern = generate_triangle_and_edge_pattern(number_of_vertices_labels)
    # else:
    #     print("error type")
    pattern_dir = "../data/large_pattern/patterns/"
    # if not os.path.isdir(pattern_dir):
    #     os.mkdir(pattern_dir)
    # pattern.write(pattern_dir + "P_" + pattern_type + "_NL" + str(number_of_vertices_labels) + "_" + str(p) + ".gml")
    pattern = ig.read(pattern_dir + "P_" + pattern_type + "_NL" + str(number_of_vertices_labels) + "_" + str(p) + ".gml")

    # for g in range(graph_random):
    # for number_of_graph_vertices in number_of_graph_vertices_list:
    if number_of_graph_vertices <= number_of_vertices_labels:
        return
    positive_counts = 0
    negative_counts = 0
    g = 0
    while positive_counts < graph_random / 2 or negative_counts < graph_random / 2:
        graph = generate_delaunay_graph(number_of_graph_vertices, number_of_vertices_labels)

        save_graph_dir_p = "../data/large_pattern/graphs/P_" + pattern_type + "_NL" + str(
            number_of_vertices_labels) + "_" + str(p) + "/"
        save_metadata_dir_p = "../data/large_pattern/metadata/P_" + pattern_type + "_NL" + str(
            number_of_vertices_labels) + "_" + str(p) + "/"
        if not os.path.exists(save_metadata_dir_p):
            os.makedirs(save_metadata_dir_p)

        graph_id = "G_N" + str(number_of_graph_vertices) + "_" + pattern_type + "_NL" + str(
            number_of_vertices_labels) + "_" + str(g)
        if not os.path.exists(save_graph_dir_p):
            os.makedirs(save_graph_dir_p)

        pc = PatternChecker()
        subisomorphisms = pc.get_subisomorphisms(graph, pattern)
        metadata = {"counts": len(subisomorphisms), "subisomorphisms": subisomorphisms}

        if len(subisomorphisms) == 0:
            negative_counts += 1
        else:
            positive_counts += 1

        if len(subisomorphisms) == 0 and negative_counts <= graph_random / 2:
            g += 1
            graph.write(os.path.join(save_graph_dir_p, graph_id + ".gml"))
            with open(os.path.join(save_metadata_dir_p, graph_id + ".meta"), "w") as f:
                json.dump(metadata, f)
        elif len(subisomorphisms) != 0 and positive_counts <= graph_random / 2:
            g += 1
            graph.write(os.path.join(save_graph_dir_p, graph_id + ".gml"))
            with open(os.path.join(save_metadata_dir_p, graph_id + ".meta"), "w") as f:
                json.dump(metadata, f)



number_of_graph_vertices_list = [128]
# number_of_vertices_labels_list = [8, 16, 32]
number_of_vertices_labels_list = [32]
pattern_type_list = ["triangle"]
pattern_random = 50
# pattern_list = [40,41,43]
graph_random = 4000

core_num = cpu_count()
pool = Pool(core_num)
for p in range(pattern_random):
    for pattern_type in pattern_type_list:
        for number_of_vertices_labels in number_of_vertices_labels_list:
            for number_of_graph_vertices in number_of_graph_vertices_list:
                pool.apply_async(generate,(pattern_type, number_of_vertices_labels,  number_of_graph_vertices, p,))
                # generate(pattern_type, number_of_vertices_labels,  number_of_graph_vertices, p)
#
pool.close()
pool.join()



# generate('adjedge', 16,  32, 1)