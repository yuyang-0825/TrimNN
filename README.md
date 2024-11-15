# TrimNN: Cellular community motif modeling for multicellular organization in complex tissues

<img src="https://img.shields.io/badge/TrimNN-v0.0.1-blue"> <img src="https://img.shields.io/badge/Platform-Linux-blue"> <img src="https://img.shields.io/badge/Language-python3-blue"> <img src="https://img.shields.io/badge/License-MIT-blue">

<p align="center">
  <img height="450" width="750" src="https://github.com/yuyang-0825/TrimNN/blob/main/TrimNN.png"/>
</p>

**Tri**angulation cellular community **m**otif **N**eural **N**etwork (**TrimNN**), a bottom-up approach to estimate the prevalence of sizeable conservative cell organization patterns as Cellular Community motifs (CC motifs) in spatial omics.

This empowered graph isomorphism network-based framework adopts inductive bias in cellular communities and focuses on estimating the relative abundance in the triangulated space. Beyond clusters of cell type composition from classical top-down multicellular neighborhood analysis, this method differentiates cellular niches as countable building blocks in recurring interconnections of various types, presenting interpretability and generalizability in cellular neighborhood analysis. In colorectal cancer and neurodegenerative disease studies using spatial proteomics and spatial transcriptomics, various sizes of CC motifs reveal diverse relations between micro-molecular spatially distributed cell types and macro phenotypical biological functions. Notably, orthologous to gene biomarkers, the identified spatial CC motifs differentiate patient survivals in colorectal cancer studies, which cannot be inferred by other existing tools. 

## System Requirements
Tested on Red Hat Enterprise Linux 7.9

### Install TrimNN from Github
```
git clone https://github.com/yuyang-0825/TrimNN
cd TrimNN
```
### Create virtual environment and install dependencies

#### Create virtual environment
```
conda create -n TrimNNEnv python=3.9 
conda activate TrimNNEnv
```
#### Install Pytorch and DGL 
* **Linux with CUDA**
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
```
pip install dgl==1.1.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
```
#### Install Other dependencies
```
pip install -r requirements.txt
```

## Data Preparation

### Input Spatial Omics Data
A spatial omics data should include ```X```, ```Y```(coordinates) and ```cell_type```  columns to generate a cellular community graph. [[example]](https://github.com/yuyang-0825/TrimNN/blob/main/demo_data/demo_data.csv)

Generate gml file from your input CSV file as TrimNN's input.
```
python csv2gml.py --path demo_data/demo_data.csv
```
* --path: The path of input data.
* The gml file with the same name will appear in the same folder.

 
## Demo

### Function 1: Identify specific size top overrepresented CC motif
To identify the specific size top overrepresented CC motif in the cellular community graph, run:
```
python specific_size.py -size 3 -k 2 -graph demo_data/demo_data.gml -celltype 8 -outpath result/
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
python all_size.py -size 4 -k 2 -graph demo_data/demo_data.gml -celltype 8 -outpath result/
```
##### Command Line Arguments:
*	-size: The maximum size of the generated top overrepresented CC motifs.[default: 5] [maximum: 9]
*	-k: k-hop.
*	-graph: file path for input cellular community graph.
*	-celltype: number of cell types.
*	-outpath: folder path for output result.
