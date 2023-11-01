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


## Model Training and Evaluation
To get the model in the paper, just run
```
cd src
python train_classification_data.py --model RGIN --predict_net DIAMNet \
    --predict_net_mem_init mean --predict_net_mem_len 4 --predict_net_recurrent_steps 3 \
    --gpu_id 0 --batch_size 256 \
    --max_npv 10  --max_npe 8 --max_npvl 32 --max_npel 8 \
    --max_ngv 128 --max_nge 1024 --max_ngvl 32 --max_ngel 16 \
    --pattern_dir data/4motif/patterns \
    --graph_dir data/4motif/graphs \
    --metadata_dir data/4motif/metadata \
    --save_data_dir data/4motif \
    --save_model_dir 4motif/RGIN-DIAMNet
```
To evaluate the model run
```
cd src
python evaluation_classify.py
```


