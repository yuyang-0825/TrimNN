from itertools import product
import random
from igraph import Graph



def generate_6order_pattern(pattern_vertices_labels):   # 6 nodes
    # pattern of triangles
    pattern = Graph(n=6, edges=[[0, 1], [0, 2], [1,2],[2,3],[2,4],[3,4],[4,5]])
    pattern.vs["label"] = pattern_vertices_labels
    return pattern


def generate_7order_pattern(pattern_vertices_labels):   # 7 nodes
    # pattern of triangles
    pattern = Graph(n=7, edges=[[0, 1], [0, 2], [1,2],[2,3],[2,4],[3,4],[4,5],[4,6],[5,6]])
    pattern.vs["label"] = pattern_vertices_labels
    return pattern

def generate_8order_pattern(pattern_vertices_labels):   # 8 nodes
    # pattern of triangles
    pattern = Graph(n=8, edges=[[0, 1], [0, 2], [1,2],[2,3],[2,4],[3,4],[4,5],[4,6],[5,6],[6,7]])
    pattern.vs["label"] = pattern_vertices_labels
    return pattern

def generate_9order_pattern(pattern_vertices_labels):   # 9 nodes
    # pattern of triangles
    pattern = Graph(n=9, edges=[[0, 1], [0, 2], [1,2],[2,3],[2,4],[3,4],[4,5],[4,6],[5,6],[6,7],[6,8],[7,8]])
    pattern.vs["label"] = pattern_vertices_labels
    return pattern


# generate_triangle_pattern(6)

def label_list(number_of_vertices_labels,repeat,order):
    numbers = list(range(number_of_vertices_labels))
    combinations = product(numbers, repeat=order)
    combinations_list = list(combinations)
    unique_combinations = list(set(tuple(sorted(combo)) for combo in combinations_list))
    random.shuffle(unique_combinations)
    unique_combinations = unique_combinations[0:repeat]
    return unique_combinations, repeat


order_list=[6,7,8,9]
repeat_number = 100
number_of_vertices_labels_list = [8, 16, 32]

for number_of_vertices_labels in number_of_vertices_labels_list:
    labels_list, repeat = label_list(number_of_vertices_labels,repeat_number,6)
    for i in range(repeat):
        labels = labels_list[i]
        pattern = generate_6order_pattern(labels)
        pattern_name = "P_6order_NL"+str(number_of_vertices_labels)+"_"+str(i)+".gml"
        pattern_path = "../data/6order/patterns/"
        pattern.write(pattern_path + pattern_name)

# for number_of_vertices_labels in number_of_vertices_labels_list:
#     labels_list, repeat = label_list(number_of_vertices_labels,repeat_number,7)
#     for i in range(repeat):
#         labels = labels_list[i]
#         pattern = generate_7order_pattern(labels)
#         pattern_name = "P_7order_NL"+str(number_of_vertices_labels)+"_"+str(i)+".gml"
#         pattern_path = "/mnt/pixstor/data/yykk3/scmotif/data/7order/patterns/"
#         pattern.write(pattern_path + pattern_name)
#
#
# for number_of_vertices_labels in number_of_vertices_labels_list:
#     labels_list, repeat = label_list(number_of_vertices_labels,repeat_number,8)
#     for i in range(repeat):
#         labels = labels_list[i]
#         pattern = generate_8order_pattern(labels)
#         pattern_name = "P_8order_NL"+str(number_of_vertices_labels)+"_"+str(i)+".gml"
#         pattern_path = "/mnt/pixstor/data/yykk3/scmotif/data/8order/patterns/"
#         pattern.write(pattern_path + pattern_name)
#
# for number_of_vertices_labels in number_of_vertices_labels_list:
#     labels_list, repeat = label_list(number_of_vertices_labels,repeat_number,9)
#     for i in range(repeat):
#         labels = labels_list[i]
#         pattern = generate_9order_pattern(labels)
#         pattern_name = "P_9order_NL"+str(number_of_vertices_labels)+"_"+str(i)+".gml"
#         pattern_path = "/mnt/pixstor/data/yykk3/scmotif/data/9order/patterns/"
#         pattern.write(pattern_path + pattern_name)