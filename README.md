# A bottom-up strategy to explore sizable cellular community motifs as building blocks of multicellular organization using TrimNN

<p align="center">
  <img height="450" width="750" src="https://github.com/yuyang-0825/TrimNN/blob/main/figure/TrimNN.png"/>
</p>

**Triangulation cellular community Motif Neural Network (TrimNN)**, an empowered bottom-up approach designed to estimate the prevalence of sizeable CC motifs in a triangulated cell graph.

## Python package Dependencies
* tqdm
* numpy
* pandas
* scipy
* tensorboardX
* argparse
* python-igraph == 0.9.6
* torch >= 1.13.1
* dgl == 1.1.2


## Identify top overrepresented network motifs
To identify the size-3 overrepresented network motifs in the triangulated graph, run:
```
cd src
python enumerate_size3.py -k 1 -graph ../graph/8/G_N64_triangle_NL8_0.gml -nodetype 8 -outpath size-3.csv
```

#### Command Line Arguments:
*	-k k-hop
*	-graph  file path for triangulated graph
*	-nodetype number of node types
*	-outpath file path for output result

If you want to get the results of the other two size-4 subgraphs， run:
```
cd src
python enumerate_size4-1.py -k 1 -graph ../graph/8/G_N64_triangle_NL8_0.gml -nodetype 8 -outpath size4-1.csv
```
and
```
cd src
python enumerate_size4-2.py -k 1 -graph ../graph/8/G_N64_triangle_NL8_0.gml -nodetype 8 -outpath size4-2.csv
```
Each size-4 result generation takes roughly 2 hours to run on an NVIDIA A100 GPU.
