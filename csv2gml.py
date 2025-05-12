import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from igraph import Graph
import os
import argparse
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer input to gml file')
    parser.add_argument('-target', type=str, default='demo_data/demo_data.csv',
                        help='The path of input graph data')
    parser.add_argument('-out', type=str, default='demo_data/demo_data.gml',
                        help='The  output path of generated gml data')
    parser.add_argument('-motif_size', type=int,
                        help='The size of input motif')
    parser.add_argument('-motif_label', type=str,help='The cell type of input motif(combine with "_")')
    parser.add_argument('-prune', type=bool,default=True, help='Whether to prune outlier edges.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_path = args.target
    out_path = args.out
    prune = args.prune
    out_folder = out_path.split('/')[0]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    input_folder = input_path.split('/')[0]
    input_name = input_path.split('/')[1]

    df=pd.read_csv(input_path)
    unqiue_cell_type = sorted(df['cell_type'].unique())

    points = np.stack((df['X'],df['Y']),axis=-1)
    tri = Delaunay(points)

    if prune==True:
        edge_lengths = []
        edge_ids = set()

        # First pass to collect unique edge lengths
        for triangle in tri.simplices:
            for i in range(3):
                i0 = triangle[i]
                i1 = triangle[(i + 1) % 3]
                if (i1, i0) in edge_ids:
                    continue
                length = np.linalg.norm(points[i0] - points[i1])
                edge_lengths.append(length)
                edge_ids.add((i0, i1))

        # Compute log-normal threshold
        log_mean = np.mean(np.log(edge_lengths))
        log_std = np.std(np.log(edge_lengths))
        threshold = stats.lognorm.ppf(0.99, loc=log_mean, s=log_std, scale=np.exp(log_mean))

        # Classify edges
        small_edges = set()
        large_edges = set()

        for edge in edge_ids:
            i0, i1 = edge
            length = np.linalg.norm(points[i0] - points[i1])
            if length < threshold:
                small_edges.add(edge)
            else:
                large_edges.add(edge)

        # Filter triangles
        small_triangles = []
        for triangle in tri.simplices:
            all_small = True
            for i in range(3):
                i0 = triangle[i]
                i1 = triangle[(i + 1) % 3]
                if (i0, i1) in large_edges or (i1, i0) in large_edges:
                    all_small = False
                    break
            if all_small:
                small_triangles.append(triangle)
        triangles = small_triangles
    else:
        triangles = tri.simplices

    adjmatrix = np.zeros((points.shape[0], points.shape[0]))
    for triangle in triangles:
        adjmatrix[triangle[0], triangle[1]] = 1
        adjmatrix[triangle[1], triangle[0]] = 1
        adjmatrix[triangle[0], triangle[2]] = 1
        adjmatrix[triangle[2], triangle[0]] = 1
        adjmatrix[triangle[1], triangle[2]] = 1
        adjmatrix[triangle[2], triangle[1]] = 1

    # generate igraph according to adj matrix
    graph = Graph.Adjacency(adjmatrix, mode='undirected')

    cell_types = df['cell_type']

    label_to_int = {label: idx for idx, label in enumerate(unqiue_cell_type)}
    label = [label_to_int[cell_type] for cell_type in cell_types]

    graph.vs["label"] = label
    graph.es["label"] = 0

    graph.write(out_path)

    if args.motif_size:
        motif_size = args.motif_size
        motif_label = args.motif_label

        if motif_size == 3:
            motif = Graph(n=3, edges=[[0, 1], [0, 2], [1, 2]])
        elif motif_size == 4:
            motif = Graph(n=4, edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
        elif motif_size == 5:
            motif = Graph(n=5, edges=[[0, 1], [0, 2], [1, 2], [0, 3], [0, 4], [3, 4]])
        elif motif_size == 6:
            motif = Graph(n=6, edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4], [4, 5], [3, 5]])
        elif motif_size == 7:
            motif = Graph(n=7, edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [5, 6]])
        elif motif_size == 8:
            motif = Graph(n=8, edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [5, 6],
                                                [5, 7], [6, 7]])
        elif motif_size == 9:
            motif = Graph(n=9, edges=[[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [6, 7],
                                                [6, 8], [7, 8]])
        motif_label = motif_label.split('_')
        motif_label_int = [label_to_int[label] for label in motif_label]
        motif.vs["label"] = motif_label_int
        motif.es["label"] = 0

        motif.write(os.path.join(out_folder, 'size-'+str(motif_size)+'.gml'))
