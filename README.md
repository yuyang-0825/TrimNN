# Exploring building blocks of cell organization by estimating network motifs using graph isomorphism network

**TrimNN**： a neural network that efficiently estimates complex cell network motifs, aids in understanding cell organization's role in biology

<p align="center">
  <img height="500" width="750" src="https://github.com/yuyang-0825/TrimNN/blob/main/figure/TrimNN.png"/>
</p>


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
