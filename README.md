# ReLU's Revival: On the Entropic Overload in Normalization-Free Large Language Models

This is the official code for our [ATTRIB@NeurIPS'24](https://attrib-workshop.cc/) workshop [paper](https://arxiv.org/abs/2410.09637).

## Repository Structure

```
.
├── normfree_transformers/
│   ├── config/
│   │   ├── config.yaml
│   │   ├── model/
│   │   │   └── default.yaml
│   │   └── train/
│   │       └── train.yaml
│   ├── model_utils.py   
│   └── train_utils.py
├── scripts/
│   ├── experiments/
│   │   ├── train_baseline.sh
│   │   └── train_normfree.sh
│   └── plotting/
│       ├── plot_headwise_entropy_heatmap.py
│       ├── plot_layerwise_entropy.py
│       └── plot_layerwise_nan_counts.py
├── requirements.txt
└── run_clm.py
```


## Overview

This repository contains the implementation of our research on normalization-free transformers and their entropic behavior. We investigate the role of ReLU activation in large language models without normalization layers.

## Installation

git clone https://github.com/[username]/relus-revival
cd relus-revival
pip install -r requirements.txt

## Project Components

### Experiments
- `scripts/experiments/`:
  - `train_baseline.sh`: Script for training baseline models
  - `train_normfree.sh`: Script for training normalization-free models

### Analysis & Visualization
- `scripts/plotting/`:
  - `plot_headwise_entropy_heatmap.py`: Generates attention head entropy heatmaps for trained networks
  - `plot_layerwise_entropy.py`: Plots layer-wise entropy analysis during pre-training phase
  - `plot_layerwise_nan_counts.py`: Plots layer-wise  NaNs for tracking training instability 
  
## Usage

To train a normalization-free model:
bash scripts/experiments/train_normfree.sh

To train a baseline model:
bash scripts/experiments/train_baseline.sh

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@article{jha2024relusrevival,
       title={ReLU's Revival: On the Entropic Overload in Normalization-Free Large Language Models},
       author={Jha, Nandan Kumar and Reagen, Brandon},
       journal={2nd Workshop on Attributing Model Behavior at Scale (NeurIPS)},
       year={2024}
}
```
## Contact

nj2049@nyu.edu
