# A bottom-up strategy to explore sizable cellular community motifs as building blocks of multicellular organization using TrimNN
<img src="https://img.shields.io/badge/TrimNN-v0.0.1-blue"> <img src="https://img.shields.io/badge/Platform-Linux-blue"> <img src="https://img.shields.io/badge/Language-python3-blue"> <img src="https://img.shields.io/badge/License-MIT-blue">

<p align="center">
  <img height="450" width="750" src="https://github.com/yuyang-0825/TrimNN/blob/main/figure/Figure.png"/>
</p>

**Triangulation cellular community Motif Neural Network (TrimNN)**, an empowered bottom-up approach designed to estimate the prevalence of sizeable CC motifs in a triangulated cell graph. Spatially resolved transcriptomics, e.g., STARmap PLUS and 10X Xenium, and spatial proteomics data, e.g., MIBI-TOF and CODEX, are used as input to generate corresponding cellular community. After process of subgraph matching and pattern growth, TrimNN estimates different Size-K overrepresented CC motifs. These CC motifs can be biologically interpreted in the downstream analysis, including statistical summarization, cellular level interpretation within cell-cell communication analysis, gene level interpretation within differentially expressed gene analysis, e.g., GO enrichment analysis and pathway enrichment analysis, and phenotypical analysis within the availability of phenotypical information, e.g., survival curve and supervised analysis. 


## System Requirements
### Install  Dependencies
``` 
 pip install -r requirements.txt
```

### Install TrimNN from GitHub
```
git clone https://yuyang-0825/TrimNN
cd TrimNN
```
## Data Preparation

### Input Spatial Omics Data
A spatial omics data should include ```X```, ```Y```(coordinates) and ```cell_type```  columns to generate a cellular community graph. [[example]](https://github.com/yuyang-0825/TrimNN/blob/main/spatial_data/demo_data.csv)

Generate gml file from your input CSV file as TrimNN's input.
```
cd src 
python csv2gml.py --path spatial_data/demo_data.csv
```
* --path: The path of input data.
* The gml file with the same name will appear in the same folder.

 
## Demo

### Function 1: Identify specific size top overrepresented CC motif
To identify the specific size top overrepresented CC motif in the cellular community graph, run:
```
python specific_size.py -size 3 -k 2 -graph spatial_data/demo_data.gml -celltype 8 -outpath result/
```
##### Command Line Arguments:
*	-size: specific size of CC motif (from 3 to 9).
*	-k: k-hop.
*	-graph: file path for input cellular community graph.
*	-celltype: number of cell types.
*	-outpath: folder path for output result.

### Function 2: Identify all top overrepresented CC motifs
To identify all top overrepresented CC motifs from size3 to size5(default) in the cellular community graph, run:
```
python all_size.py -size 5 -k 2 -graph spatial_data/demo_data.gml -celltype 8 -outpath result/
```
##### Command Line Arguments:
*	-size: The maximum size of the generated top overrepresented CC motifs.[default: 5] [maximum: 9]
*	-k: k-hop.
*	-graph: file path for input cellular community graph.
*	-celltype: number of cell types.
*	-outpath: folder path for output result.
