# A bottom-up strategy to explore sizable cellular community motifs as building blocks of multicellular organization using TrimNN
<img src="https://img.shields.io/badge/TrimNN-v1.0.0-blue"> <img src="https://img.shields.io/badge/Platform-Linux-blue"> <img src="https://img.shields.io/badge/Language-python3-blue"> <img src="https://img.shields.io/badge/License-MIT-blue">

<p align="center">
  <img height="450" width="750" src="https://github.com/yuyang-0825/TrimNN/blob/main/figure/Figure.png"/>
</p>

**Triangulation cellular community Motif Neural Network (TrimNN)**, an empowered bottom-up approach designed to estimate the prevalence of sizeable CC motifs in a triangulated cell graph. Spatially resolved transcriptomics, e.g., STARmap PLUS and 10X Xenium, and spatial proteomics data, e.g., MIBI-TOF and CODEX, are used as input to generate corresponding cellular community. After process of subgraph matching and pattern growth, TrimNN estimates different Size-K overrepresented CC motifs. These CC motifs can be biologically interpreted in the downstream analysis, including statistical summarization, cellular level interpretation within cell-cell communication analysis, gene level interpretation within differentially expressed gene analysis, e.g., GO enrichment analysis and pathway enrichment analysis, and phenotypical analysis within the availability of phenotypical information, e.g., survival curve and supervised analysis. 


## System Requirements
#### Python Dependencies
``` 
 tqdm
 numpy
 pandas
 scipy
 tensorboardX
 argparse
 python-igraph == 0.9.6
 torch >= 1.13.1
 dgl == 1.1.2
```

### Install TrimNN from GitHub
```
git clone https://yuyang-0825/TrimNN
cd TrimNN
```
## Data Preparation
### Input Spatial Omics Data
A spatial omics data should include x,y coordinates and cell types to generate cellular community graph. It is used for generating corresponding size-k overrepresented CC motifs. [[example]](https://github.com/yuyang-0825/TrimNN/blob/main/spatial_data/demo_data.csv)

Generate gml file from your input csv file as TrimNN's input.
```
cd src 
python csv2gml.py --path spatial_data/demo_data.csv
```

## Demo

### Identify top overrepresented network motifs
To identify the size-3 overrepresented network motifs in the triangulated graph, run:
```
cd src
python enumerate_size3.py -k 1 -graph ../graph/8/G_N64_triangle_NL8_0.gml -nodetype 8 -outpath size-3.csv
```

##### Command Line Arguments:
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
