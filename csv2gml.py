import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from igraph import Graph
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Transfer input to gml file')
    parser.add_argument('--path', type=str, default='demo_data/demo_data.csv',
                        help='The path of input data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    input_path = args.path
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

