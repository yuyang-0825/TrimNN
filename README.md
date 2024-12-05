# TrimNN: Characterizing cellular community motifs for studying multicellular topological organization in complex tissues

<img src="https://img.shields.io/badge/TrimNN-v0.0.1-blue"> <img src="https://img.shields.io/badge/Platform-Linux-blue"> <img src="https://img.shields.io/badge/Language-python3-blue"> <img src="https://img.shields.io/badge/License-MIT-blue">

<p align="center">
  <img height="450" width="800" src="https://github.com/yuyang-0825/TrimNN/blob/main/TrimNN.png"/>
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
* **Linux with cpu only**
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```
```
pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/repo.html
```
#### Install Other dependencies
```
pip install -r requirements.txt
```
* Note: If you encounter this problem ```undefined symbol: iJIT_NotifyEvent``` when running, please try ```conda install mkl==2024.0``` to solve.
## Data Preparation

### Input Spatial Omics Data
A spatial omics data should include ```X```, ```Y```(coordinates) and ```cell_type```  columns to generate a cellular community graph. [[example]](https://github.com/yuyang-0825/TrimNN/blob/main/demo_data/demo_data.csv)

Generate gml file from your input CSV file as TrimNN's input.
```
python csv2gml.py -target demo_data/demo_data.csv -out demo_data/demo_data.gml
```
* -target: The path of input target cellular community graph data.
* -out: The path of generated gml data

**Optional:** If you want to input a specific CC motif to test the "Subgraph Matching" function:
```
python csv2gml.py -target demo_data/demo_data.csv -out demo_data/demo_data.gml -motif_size 3 -motif_label Micro_Micro_Micro
```
* -target: The path of input target cellular community graph data.
* -out: The path of generated target gml data
* -motif_size: The size of input motif.
* -motif_label: The cell type of input motif(combine with "_").
* The generated specific motif gml will appear in the same folder as target graph gml.
 
## Demo
### Function 1: Subgraph Matching
To predict the number of occurrences of the input CC motif in the target cellular community graph, run:
```
python TrimNN.py -function subgraph_matching -motif demo_data/size-3.gml -k 2 -target demo_data/demo_data.gml -outpath result_function1/
```
##### Command Line Arguments:
*	-motif: The file path for input CC motif.
*	-k: Use k-hop to get each node’s enclosed graph(here k=2 is the default value).
*	-target: The file path for input cellular community graph.
*	-outpath: You should expected predicted occurrence number in this folder.
*	This function takes about 1 minute to generate result on the machine with A100 GPU.
  
### Function 2: Identify specific size top overrepresented CC motif
To identify the specific size top overrepresented CC motif in the target cellular community graph, run:
```
python TrimNN.py -function specific_size -size 3 -k 2 -target demo_data/demo_data.gml -celltype 8 -outpath result_function2/
```
##### Command Line Arguments:
*	-size: The specific size of CC motif (from 3 to 9).
*	-k: Use k-hop to get each node’s enclosed graph(here k=2 is the default value).
*	-target: The file path for input cellular community graph.
*	-celltype: The number of cell types in the input target gml(The input demo_data.gml here has 8 cell types).
*	-outpath: You should expected two files, one file is .gml file is top overrepresented CC motif, the other file is all specific size motif's predicted occurrence number in this folder.
*	This function takes about 3 minutes to generate result on the machine with A100 GPU.

### Function 3: Identify all top overrepresented CC motifs
To identify all top overrepresented CC motifs from size3 to size4(default) in the target cellular community graph, run:
```
python TrimNN.py -function all_size -size 4 -k 2 -target demo_data/demo_data.gml -celltype 8 -outpath result_function3/ -search greedy 
```
##### Command Line Arguments:
*	-size: The maximum size of the generated top overrepresented CC motifs (from 3 to 9).
*	-k: Use k-hop to get each node’s enclosed graph(here k=2 is the default value).
*	-target: The file path for input cellular community graph.
*	-celltype: The number of cell types in the input target gml(The input demo_data.gml here has 8 cell types).
*	-outpath: You should expected .gml files of top overrepresented CC motif from size-3 to specified size(here is 4) in this folder.
*	-search: Search method for motif growth, now is greedy.
*	This function takes about 3 minutes to generate result on the machine with A100 GPU.
