# TRAR-VQA
This is an official implement for ICCV 2021 paper ["TRAR: Routing the Attention Spans in Transformers for Visual Question Answering"](). It currently includes the code for training TRAR on VQA2.0

## Updates
2021/8/13
- The project page for TRAR is avaliable.

## Introduction
**TRAR vs Standard Transformer**
<p align="center">
	<img src="misc/trar_block.png" width="550">
</p>

**TRAR Overall**
<p align="center">
	<img src="misc/trar_overall.png" width="550">
</p>

## Table of Contents
0. [Installation](#Installation)
1. [Dataset setup](#Dataset-setup)
2. [Config Introduction](#Config-Introduction)
3. [Training](#Training)
4. [Validation and Testing](#Validation-and-Testing)

### Installation
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
- Install [Spacy](https://spacy.io/) and initialize the [GloVe]() as follows:
```bash
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```

### Dataset setup
see [DATA.md](https://github.com/rentainhe/TRAR-VQA/blob/main/DATA.md)

### Config Introduction
In [trar.yml]("configs/vqa/trar.yml) config we have these specific settings for `TRAR` model
```
ORDERS: [0, 1, 2, 3]
IMG_SCALE: 8 
ROUTING: 'hard' # {'soft', 'hard'}
ROUTING_MODE: 'attention' # {'attention', 'avg'}
TAU_POLICY: 1 # {0: 'SLOW', 1: 'FAST', 2: 'FINETUNE'}
TAU_MAX: 10
TAU_MIN: 0.1
```
- `ORDERS=list`, to set the local attention window size for routing.`0` for global attention.
- `IMG_SCALE=int`, which should be equal to the `image feature size` used for training. You should set `IMG_SCALE: 16` for `16 × 16` training features.
- `ROUTING={'hard', 'soft'}`, to set the `Routing Block Type` in TRAR model.
- `ROUTING_MODE={'attention', 'avg'}`, to set the `Downsample Strategy` used in `Routing Block`.
- `TAU_POLICY={0, 1, 2}`, to set the `temperature schedule` in training TRAR when using `ROUTING: 'hard'`.
- `TAU_MAX=int`, to set the maximum temperature in training.
- `TAU_MIN=int`, to set the minimum temperature in training.

**`TAU_POLICY` visualization**

For `MAX_EPOCH=13` with `WARMUP_EPOCH=3` we have the following policy strategy:
<p align="center">
	<img src="misc/policy_visualization.png" width="550">
</p>

### Training
**Train model on VQA-v2 with default hyperparameters:**
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar'
```
and the training log will be seved to:
```
results/log/log_run_<VERSION>.txt
```
Args:
- `--DATASET={'vqa', 'clevr'}` to choose the task for training
- `--GPU=str`, e.g. `--GPU='2'` to train model on specific GPU device.
- `--SPLIT={'train', 'train+val', train+val+vg'}`, which combines different training datasets. The default training split is `train`.
- `--MAX_EPOCH=int` to set the total training epoch number.


**Resume Training**
Resume training from specific saved model weights
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --RESUME=True --CKPT_V=str --CKPT_E=int
```
- `--CKPT_V=str`: the specific checkpoint version
- `--CKPT_E=int`: the resumed epoch number

**Multi-GPU Training and Gradient Accumulation**
1. Multi-GPU Training:
Add `--GPU='0, 1, 2, 3...'` after the training scripts.
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --GPU='0,1,2,3'
```
The batch size on each GPU will be divided into `BATCH_SIZE/GPUs` automatically.

2. Gradient Accumulation:
Add `--ACCU=n` after the training scripts
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --ACCU=2
```
This makes the optimizer accumulate gradients for `n` mini-batches and update the model weights once. `BATCH_SIZE` should be divided by `n`.

### Validation and Testing
**Warning**: The args `--MODEL` and `--DATASET` should be set to the same values as those in the training stage.

**Validate on Local Machine**
Offline evaluation only support the evaluations on the `coco_2014_val` dataset now.
1. Use saved checkpoint
```bash
python3 run.py --RUN='val' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

2. Use the absolute path
```bash
python3 run.py --RUN='val' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_PATH=str
```

**Online Testing**
All the evaluations on the `test` dataset of VQA-v2 and CLEVR benchmarks can be achieved as follows:
```bash
python3 run.py --RUN='test' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

Result file are saved at:

`results/result_test/result_run_<CKPT_V>_<CKPT_E>.json`


## Citing TRAR
