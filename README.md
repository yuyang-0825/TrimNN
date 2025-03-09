# TrimNN: Characterizing cellular community motifs for studying multicellular topological organization in complex tissues

<img src="https://img.shields.io/badge/TrimNN-v0.0.1-blue"> <img src="https://img.shields.io/badge/Platform-Linux-blue"> <img src="https://img.shields.io/badge/Language-python3-blue"> <img src="https://img.shields.io/badge/License-MIT-blue">

<p align="center">
  <img height="450" width="800" src="https://github.com/yuyang-0825/TrimNN/blob/main/TrimNN.png"/>
</p>

The spatial arrangement of cells plays a pivotal role in shaping tissue functions in various biological systems and diseased microenvironments. However, it is still under-investigated of the topological coordinating rules among different cell types as tissue spatial patterns. Here, we introduce the **Tri**angulation cellular community **m**otif **N**eural **N**etwork (**TrimNN**), a bottom-up approach to estimate the prevalence of sizeable conservative cell organization patterns as Cellular Community (CC) motifs in spatial transcriptomics and proteomics. Different from clustering cell type composition from classical top-down analysis, TrimNN differentiates cellular niches as countable topological blocks in recurring interconnections of various types, representing multicellular neighborhoods with interpretability and generalizability. This graph-based deep learning framework adopts inductive bias in CCs and uses a semi-divide and conquer approach in the triangulated space. In spatial omics studies, various sizes of CC motifs identified by TrimNN robustly reveal relations between spatially distributed cell-type patterns and diverse phenotypical biological functions. 

## System Requirements
Tested on Red Hat Enterprise Linux 7.9 and SUSE Linux Enterprise Server 15 SP3.

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
#### Install PyTorch and DGL 
* *Note : If you encounter this problem with an ```undefined symbol: iJIT_NotifyEvent``` when running, please try ```conda install mkl==2024.0``` to solve.
* **Linux with CUDA**
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
```
pip install dgl==1.1.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
```
* **Linux with CPU only**
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
## Data Preparation

### Input Spatial Omics Data
A spatial omics data should include ```X```, ```Y```(coordinates) and ```cell_type```  columns to generate a cellular community graph. [[example]](https://github.com/yuyang-0825/TrimNN/blob/main/demo_data/demo_data.csv)

Generate a gml file from your input CSV file as TrimNN's input.
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
* -motif_size: The size of the input motif.
* -motif_label: The cell type of input motif (combined with "_").
* The generated specific motif gml will appear in the same folder as the target graph gml.
 
## Demo
### Function 1: Subgraph Matching
To predict the number of occurrences of the input CC motif in the target cellular community graph, run:
```
python TrimNN.py -function subgraph_matching -motif demo_data/size-3.gml -k 2 -target demo_data/demo_data.gml -outpath result_function1/
```
##### Command Line Arguments:
*	-motif: The file path for input CC motif.
*	-k: Use k-hop to get each node’s enclosed graph (here k=2 is the default value).
*	-target: The file path for input cellular community graph.
*	-outpath: Users should expect one file containing the predicted occurrence number of the input CC motif in this folder.
*	This function takes about 1 minute to generate results on the machine with A100 GPU.
  
### Function 2: Identify specific size top overrepresented CC motifs
To identify the specific size of top overrepresented CC motifs in the target cellular community graph, run:
```
python TrimNN.py -function specific_size -size 3 -k 2 -target demo_data/demo_data.gml -celltype 8 -outpath result_function2/
```
##### Command Line Arguments:
*	-size: The specific size of CC motifs (from 3 to 9).
*	-k: Use k-hop to get each node’s enclosed graph (here k=2 is the default value).
*	-target: The file path for input cellular community graph.
*	-celltype: The number of cell types in the input target gml (The input demo_data.gml here has 8 cell types).
*	-outpath: Users should expect two files in this folder, one file is .gml file of the top overrepresented CC motif, and the other .csv file contains all specific size motifs' predicted occurrence number (The first column is motifs in igraph form, contains edge relationships between nodes. Second Column is cell type for each node. Third column is predicted occurrence number).
*	This function takes about 3 minutes to generate results on the machine with A100 GPU.
*	
**Optional:** If you want to use VF2 method to test this function:
```
python vf2_analysis.py -size 3 -target demo_data/demo_data.gml -celltype 8 -outpath result_vf2_function2/
```
*	-size: The specific size of CC motifs (from 3 to 9).
*	-target: The file path for input cellular community graph.
*	-celltype: The number of cell types in the input target gml (The input demo_data.gml here has 8 cell types).
*	-outpath: Users should expect two files in this folder, one file is .gml file of the top overrepresented CC motif, and the other .csv file contains all specific size motifs' predicted occurrence number (The first column is motifs in igraph form, contains edge relationships between nodes. Second Column is cell type for each node. Third column is predicted occurrence number).

### Function 3: Identify all top overrepresented CC motifs
To identify all top overrepresented CC motifs from size3 to size4(default) in the target cellular community graph, run:
```
python TrimNN.py -function all_size -size 4 -k 2 -target demo_data/demo_data.gml -celltype 8 -outpath result_function3/ -search greedy 
```
##### Command Line Arguments:
*	-size: The maximum size of the generated top overrepresented CC motifs (from 3 to 9).
*	-k: Use k-hop to get each node’s enclosed graph (here k=2 is the default value).
*	-target: The file path for input cellular community graph.
*	-celltype: The number of cell types in the input target gml (The input demo_data.gml here has 8 cell types).
*	-outpath: Users should expect several .gml files of top overrepresented CC motifs from size-3 to specified size (here is 4), and other .csv files contain all different sizes (from 3 to specified size (here is 4)) motifs' predicted occurrence number like Function2 in this folder.
*	-search: Search CC motifs in larger size in the process of pattern growth, currently we support greedy search.
*	This function takes about 3 minutes to generate results on the machine with A100 GPU.
