# ReLU's Revival: On the Entropic Overload in Normalization-Free Large Language Models

Welcome to the official repository for our paper, **"ReLU's Revival: On the Entropic Overload in Normalization-Free Large Language Models"**, presented at the [ATTRIB@NeurIPS'24](https://attrib-workshop.cc/) workshop. Our paper is available on [arXiv](https://arxiv.org/abs/2410.09637).

## 📝 Abstract

LayerNorm is a critical component in modern large language models (LLMs) for stabilizing training and ensuring smooth optimization. However, it introduces significant challenges in mechanistic interpretability, outlier feature suppression, faithful signal propagation, and the computational and communication complexity of private inference. This work explores desirable activation functions in normalization-free decoder-only LLMs. Contrary to the conventional preference for GELU in transformer-based models, our empirical findings demonstrate an _opposite trend_—ReLU significantly outperforms GELU in LayerNorm-free models, leading to an **8.2%** perplexity improvement. 

We discover a key issue with GELU, where early layers experience entropic overload, resulting in the under-utilization of the representational capacity of attention heads. This highlights that smoother activations like GELU are _ill-suited_ for LayerNorm-free architectures, whereas ReLU's geometrical properties—specialization in input space and intra-class selectivity—lead to improved learning dynamics and better information retention in the absence of LayerNorm. This study offers key insights for optimizing transformer architectures where LayerNorm introduces significant challenges.

## 📚 Overview

This repository contains the codebase and resources for exploring the entropic characteristics of normalization-free transformers, particularly focusing on the use of ReLU activation in large language models (LLMs). Our approach sheds light on the advantages of removing normalization layers for efficient model training and inference in transformer architectures, making this research particularly relevant for efficient private inference tasks.

## 🔧 Installation

To get started with the repository, clone the repository and install dependencies as follows:

```bash
git clone https://github.com/[username]/relu-revival-normfree
cd relu-revival-normfree
pip install -r requirements.txt

.
├── normfree_transformers/        # Core model and utility files
│   ├── config/                   # Configuration files for model and training
│   │   ├── config.yaml           # General configuration file
│   │   ├── model/
│   │   │   └── default.yaml      # Default model configuration
│   │   └── train/
│   │       └── train.yaml        # Training-specific configuration
│   ├── model_utils.py            # Model architecture utilities
│   └── train_utils.py            # Training utilities
├── scripts/
│   ├── experiments/              # Experimentation scripts for model training
│   │   ├── train_baseline.sh     # Script for training baseline model
│   │   └── train_normfree.sh     # Script for training normalization-free model
│   └── plotting/                 # Visualization and analysis scripts
│       ├── plot_headwise_entropy_heatmap.py # Head-wise entropy heatmap generator
│       ├── plot_layerwise_entropy.py        # Layer-wise entropy analysis plotter
│       └── plot_layerwise_nan_counts.py     # Tracks NaNs in layers for instability detection
├── requirements.txt              # Dependencies
└── run_clm.py                    # Main script for running LLM training



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
