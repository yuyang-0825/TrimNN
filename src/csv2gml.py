import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from igraph import Graph
import os

current_file_path = os.path.abspath(__file__)

parent_directory = os.path.dirname(current_file_path)

data_directory = os.path.join(parent_directory, '..', 'spatial_data')

file_path = os.path.join(data_directory, 'demo_data.csv')

df=pd.read_csv(file_path)
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

graph.write( os.path.join(data_directory, 'demo_data.gml'))

