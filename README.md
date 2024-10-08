# A bottom-up strategy to explore sizable cellular community motifs as building blocks of multicellular organization using TrimNN

<p align="center">
  <img height="500" width="750" src="https://github.com/yuyang-0825/TrimNN/blob/main/figure/figure1.png"/>
</p>

**Triangulation cellular community Motif Neural Network (TrimNN)**, an empowered bottom-up approach designed to estimate the prevalence of sizeable CC motifs in a triangulated cell graph. Spatially resolved transcriptomics, e.g., STARmap PLUS and 10X Xenium, and spatial proteomics data, e.g., MIBI-TOF and CODEX, are used as input to generate corresponding cellular community. After process of subgraph matching and pattern growth, TrimNN estimates different Size-K overrepresented CC motifs.  B. These CC motifs can be biologically interpreted in the downstream analysis, including statistical summarization, cellular level interpretation within cell-cell communication analysis, gene level interpretation within differentially expressed gene analysis, e.g., GO enrichment analysis and pathway enrichment analysis, and phenotypical analysis within the availability of phenotypical information, e.g., survival curve and supervised analysis. 


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
