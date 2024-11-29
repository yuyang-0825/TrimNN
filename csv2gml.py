import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from igraph import Graph
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer input to gml file')
    parser.add_argument('-graph', type=str, default='demo_data/demo_data.csv',
                        help='The path of input graph data')
    parser.add_argument('--motif_size', type=int, default=3,
                        help='The size of input motif')
    parser.add_argument('--motif_label', type=str, default="Micro&Micro&Micro" ,help='The cell type of input motif(combine with "&")')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_path = args.graph
    input_folder = input_path.split('/')[0]
    input_name = input_path.split('/')[1]

    df=pd.read_csv(input_path)
    unqiue_cell_type = sorted(df['cell_type'].unique())



    points = np.stack((df['X'],df['Y']),axis=-1)
    tri = Delaunay(points)

    adjmatrix = np.zeros((points.shape[0], points.shape[0]))
    for triangle in tri.simplices:
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

    graph.write(os.path.join(input_folder, input_name.split('.')[0] + '.gml') )

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
        motif_label = motif_label.split('&')
        motif_label_int = [label_to_int[label] for label in motif_label]
        motif.vs["label"] = motif_label_int
        motif.es["label"] = 0

        motif.write(os.path.join(input_folder, 'size-'+str(motif_size)+'.gml'))


