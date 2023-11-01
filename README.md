# Exploring building blocks of cell organization by estimating network motifs using graph isomorphism network

**TrimNN**： a neural network that efficiently estimates complex cell network motifs, aids in understanding cell organization's role in biology

<p align="center">
  <img height="500" width="750" src="https://github.com/yuyang-0825/TrimNN/blob/main/figure/TrimNN_figure.png"/>
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
To identify the overrepresented network motifs in the triangulated graph, run 
```
cd src
python enumerate_size3.py -k 1 -graph ../graph/8/G_N64_triangle_NL8_0.gml -nodetype 8 -outpath size-3.csv
python enumerate_size4-1.py -k 1 -graph ../graph/8/G_N64_triangle_NL8_0.gml -nodetype 8 -outpath size4-1.csv
python enumerate_size4-2.py -k 1 -graph ../graph/8/G_N64_triangle_NL8_0.gml -nodetype 8 -outpath size4-2.csv
```
