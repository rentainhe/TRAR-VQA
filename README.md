# TRAR-VQA
This is an official implement for ICCV 2021 paper ["TRAR: Routing the Attention Spans in Transformers for Visual Question Answering"](). It currently includes the code for training TRAR on VQA2.0

## Updates


## Introduction

## Usage
### Install
- Clone this repo
```bash
git clone https://github.com/rentainhe/TRAR-VQA.git
cd TRAR-VQA
```

- Create a conda virtual environment and activate it
```bash
conda create -n trar python=3.7 -y
conda activate trar
```

- Install `CUDA==10.1` with `cudnn7` following the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `Pytorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```
- Install other requirements:
```bash
pip install -r requirements.txt
```

### Data preparation
see [DATA.md](https://github.com/rentainhe/TRAR-VQA/blob/main/DATA.md)

## Citing TRAR