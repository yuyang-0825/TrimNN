import igraph as ig
from igraph import Graph
import numpy as np
import os
import json
from delaunay_pattern_generator import generate_hinge_pattern, generate_triangle_pattern, generate_adjacent_edge_triangles_pattern, generate_adjacent_node_triangles_pattern, generate_triangle_and_edge_pattern
from delaunay_graph_generator import generate_delaunay_graph
from pattern_checker import PatternChecker

number_of_graph_vertices_list = [8, 16, 32, 64, 128]
number_of_vertices_labels_list = [2, 4, 8, 16]
pattern_type_list = ["triangle", "hinge", "adjedge", "adjnode", "triedge"]

# number_of_graph_vertices_list = [8]
# number_of_vertices_labels_list = [2]
# pattern_type_list = ["triangle"]

pattern_random = 10
graph_random = 200


for p in range(pattern_random):
    for pattern_type in pattern_type_list:
        for number_of_vertices_labels in number_of_vertices_labels_list:
            if pattern_type == "triangle":
                pattern = generate_triangle_pattern(number_of_vertices_labels)
            elif pattern_type == "hinge":
                pattern = generate_hinge_pattern(number_of_vertices_labels)
            elif pattern_type == "adjedge":
                pattern = generate_adjacent_edge_triangles_pattern(number_of_vertices_labels)
            elif pattern_type == "adjnode":
                pattern = generate_adjacent_node_triangles_pattern(number_of_vertices_labels)
            elif pattern_type == "triedge":
                pattern = generate_triangle_and_edge_pattern(number_of_vertices_labels)
            else:
                print("error type")
            pattern.write("patterns/P_"+pattern_type+"_NL"+str(number_of_vertices_labels)+"_"+str(p)+".gml")
            for g in range(graph_random):
                for number_of_graph_vertices in number_of_graph_vertices_list:
                    if number_of_graph_vertices < number_of_vertices_labels:
                        continue
                    graph = generate_delaunay_graph(number_of_graph_vertices, number_of_vertices_labels)
                    save_graph_dir_p = "graphs/P_" + pattern_type+"_NL"+str(number_of_vertices_labels) +"_"+str(p)+ "/"
                    graph_id = "G_N"+str(number_of_graph_vertices)+"_"+pattern_type+"_NL"+str(number_of_vertices_labels)+"_"+str(g)
                    if not os.path.exists(save_graph_dir_p):
                        os.makedirs(save_graph_dir_p)
                    graph.write(os.path.join(save_graph_dir_p,graph_id+".gml"))

                    pc = PatternChecker()
                    subisomorphisms = pc.get_subisomorphisms(graph, pattern)
                    metadata = {"counts": len(subisomorphisms), "subisomorphisms": subisomorphisms}

                    save_metadata_dir_p = "metadata/P_" + pattern_type + "_NL" + str(number_of_vertices_labels) +"_"+str(p) + "/"
                    if not os.path.exists(save_metadata_dir_p):
                        os.makedirs(save_metadata_dir_p)
                    with open(os.path.join(save_metadata_dir_p, graph_id + ".meta"), "w") as f:
                        json.dump(metadata, f)





