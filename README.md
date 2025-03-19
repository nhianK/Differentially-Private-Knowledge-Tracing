##Requirements



## Models Overview

This repository contains four knowledge tracing models implemented with Differential Privacy:
- MonaCoBERT
- DKT
- BKT
- DKT-plus

Each model has its own specific command line arguments and usage instructions detailed below.

## Installation
# Install PyTorch 1.12.0
```bash
pip install torch==1.12.0 
torchvision==0.13.0
```

# Install Opacus 1.3.0 for differential privacy
```bash
pip install opacus==1.3.0
```
# Additional dependencies
pip install numpy pandas tqdm matplotlib

# Run in command line
## Usage

To train the MonacoBERT model on the ASSIST2017 dataset:

```bash
python -u train.py --model_fn model.pth --model_name monacobert --dataset_name assist2017_pid
```
To train DKT
```bash
python -u train.py --model_name=dkt --dataset_name ASSIST2012
```


