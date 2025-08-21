import pandas as pd
import argparse
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import os

def sort_motif_label(label: str) -> str:
    parts = label.split("_")
    parts_sorted = sorted(parts)
    return "_".join(parts_sorted)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Transfer input to gml file')
    parser.add_argument('-target', type=str, default='demo_data/demo_data.csv',
                        help='The path of input graph data')
    parser.add_argument('-outpath', type=str, default='visualization/',
                        help='The  output path of visualized result')
    parser.add_argument('-motif_size', type=int,default=2,
                        help='The size of input motif')
    parser.add_argument('-motif_label',default='CTX-Ex_CTX-Ex', type=str,help='The cell type of input motif(combine with "_")')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    graph_path = args.target
    motif_size = args.motif_size
    motif_label = args.motif_label
    df = pd.read_csv(graph_path)
    out_path = args.outpath
    out_folder = out_path.split('/')[0]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    points = np.stack((df['X'], df['Y']), axis=-1)
    cell_type = df['cell_type']

    unique_cell_types = df['cell_type'].unique()

    palette = sns.color_palette("hls", len(unique_cell_types))

    color_map = dict(zip(unique_cell_types, palette))

    tri = Delaunay(points)

    if motif_size > 1:
        motif_label = sort_motif_label(motif_label)

    if motif_size == 1 or motif_size == 2 or motif_size == 3:
        plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='grey', markersize=0.2, linewidth=0.1)

        # plt.show()
        lines = []
        for sim in tri.simplices:
            mtf_name = '_'.join(sorted(cell_type[sim]))

            if motif_size == 1:
                for i in sim:
                    if cell_type[i] == motif_label:

                        plt.plot(points[i, 0], points[i, 1], '.', color=color_map[cell_type[i]], markersize=2)

            elif motif_size == 2:
                edges = ['_'.join(sorted(cell_type[[sim[i],sim[(i+1)%3]]])) for i in range(3)]
                edge_idx = [[sim[i],sim[(i+1)%3]] for i in range(3)]
                for i in range(len(edges)):
                    if edges[i]==motif_label:
                        lines.append(edge_idx[i])

            elif motif_size == 3:
                if mtf_name == motif_label:
                    lines.append([sim[0],sim[1]])
                    lines.append([sim[1],sim[2]])
                    lines.append([sim[0],sim[2]])

        for pair in lines:
            node1, node2 = pair
            # Plot line between the two points
            plt.plot([points[node1, 0], points[node2, 0]], [points[node1, 1], points[node2, 1]], color='blue',
                     linewidth=0.8)
            plt.plot(points[node1, 0], points[node1, 1], '.', color=color_map[cell_type[node1]], markersize=2)
            plt.plot(points[node2, 0], points[node2, 1], '.', color=color_map[cell_type[node2]], markersize=2)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=ct,
                   markerfacecolor=color_map[ct], markersize=5)
            for ct in unique_cell_types
        ]

        plt.legend(handles=legend_elements, title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(out_folder, 'size-' + str(motif_size) + '_visualization.png'), dpi=600,
                    bbox_inches='tight')

    else:
        print("For higher-dimensional motifs, due to their structural diversity, users can customize the visualization based on the specific motif patterns of interest.")


