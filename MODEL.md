## Model
We provide our pretrained models and training log here.

## VQA-v2
We provide three groups of results (including the accuracies of Overall, Yes/No, Number and Other) for TRAR using different hyper-parameters.

- **Train -> Val:** trained on the `train` split and evaluated on the `val` split.
- **Train+val -> Test-dev:** trained on the `train+val` splits and evaluated on the `test-dev` split.
- **Train+val+vg -> Test-dev:** trained on the `train+val+vg` splits and evaluated on the `test-dev` split.

**Note that for one model, the used hyper-parameters may be different, you should modify this setting in the config file to reproduce the results.**

### Train -> Val
**TRAR Config File:** [trar.yml](configs/vqa/trar.yml)

**Note that the `BINARIZE` will not influence the training, it only changes the evaluate behavior in TRAR when using `ROUTING='hard'`, so the `BINARIZE=True` and `BINARIZE=False` share the same weight. And `POLICY` and `BINARIZE` is deprecated when `ROUTING='soft'`.**

**Pretrained Model on `8 * 8` Size Train Set**
| Model    | Base lr | ORDERS      | ROUTING    | POOLING     | POLICY    | BINARIZE |Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download             |
|:--------:|:-------:|:-----------:|:----------:|:-----------:|:---------:|:--------:|:----------:|:----------:|:----------:|:---------:|:-------------------: |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| hard       | attention   | 1         | False    | 67.61      | 85.22      | 49.66      | 58.97     | [OneDrive](https://1drv.ms/f/s!Ary9y5k2nMUxhUNnPf0VnhX-eDW5) \| [BaiduYun](https://pan.baidu.com/s/1xmtvJRhZPhGnRjf5jtGDqA) `code:v3t8` |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| hard       | attention   | 1         | True     | **67.62**  | 85.19      | 49.75      | 58.98     |           -          |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| hard       | avg         | 1         | False    | 67.59      | 85.31      | 49.85      | 58.81     | [OneDrive](https://1drv.ms/f/s!Ary9y5k2nMUxhUnK6V5D_QrERNYH) \| [BaiduYun](https://pan.baidu.com/s/16WZdO67_A94IpuMkiqp-Pg) `code:5g22` |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| hard       | avg         | 1         | True     | 67.58      | 85.30      | 49.51      | 58.78     |           -          |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| soft       | attention   | -         | False    | 67.45      | 85.03      | 49.80      | 58.75     |           -          |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| soft       | avg         | -         | False    | **67.62**  | 85.36      | 49.89      | 58.83     | [OneDrive](https://1drv.ms/f/s!Ary9y5k2nMUxhUbSCRX7m_4mZnlA) \| [BaiduYun](https://pan.baidu.com/s/1X2rCIAJiyQXRuZysaNRNwg) `code:fhgu` |


### Train+val -> Test-dev
**We've observed that TRAR with `ROUTING='soft'`, `POOLING='avg'`, `BINARIZE=False` is a bit more stable. So We trained our model on split `train+val` under these settings**

**Pretrained Model on `8 * 8` Size Train Set**
| Model    | Base lr | ORDERS      | ROUTING    | POOLING     | POLICY    | BINARIZE |Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download             |
|:--------:|:-------:|:-----------:|:----------:|:-----------:|:---------:|:--------:|:----------:|:----------:|:----------:|:---------:|:-------------------: |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| soft       | avg         | -         | False    | **71.21**  | 85.35      | 53.13      | 61.53     | [OneDrive](https://1drv.ms/f/s!Ary9y5k2nMUxhVX_aC1pEN4HAzTB) \| [BaiduYun](https://pan.baidu.com/s/1nCjnM_-jzUdJMJ94q3rlqg) `code:kwvv` |


### Train+val+vg -> Test-dev
**Pretrained Model on `8 * 8` Size Train Set**
| Model    | Base lr | ORDERS      | ROUTING    | POOLING     | POLICY    | BINARIZE |Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download             |
|:--------:|:-------:|:-----------:|:----------:|:-----------:|:---------:|:--------:|:----------:|:----------:|:----------:|:---------:|:-------------------: |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| soft       | avg         | -         | False    | **72.01**  | 87.52      | 55.06      | 62.59     | [OneDrive](https://1drv.ms/f/s!Ary9y5k2nMUxhVX_aC1pEN4HAzTB) \| [BaiduYun](https://pan.baidu.com/s/1nCjnM_-jzUdJMJ94q3rlqg) `code:kwvv` |

### Eval Example on VQA2.0

For validating `TRAR` model pretrained on `train` split under `ORDERS=[0, 1, 2, 3]`, `ROUTING='hard'`, `POOLING='attention'`, `BINARIZE=False`:
1. Download the pretrained weight here: [model](https://1drv.ms/f/s!Ary9y5k2nMUxhUNnPf0VnhX-eDW5).
2. Place the weight `epoch13.pkl` in any folder you like
3. Check the [trar.yml](configs/vqa/trar.yml) config file:
```
...
ORDERS: [0, 1, 2, 3]
IMG_SCALE: 8
ROUTING: 'hard' 
POOLING: 'attention'
...
BINARIZE: False
...
```
4. Run the following scripts:
```bash
$ cd TRAR-VQA
$ python3 run.py --DATASET vqa --MODEL trar --RUN val --CKPT_PATH /path/to/epoch13.pkl
```
**Note that you should make sure the hyper-parameters in [trar.yml](configs/vqa/trar.yml) are as the same as the pretrained model weight. You can check the hyper-parameters in the downloaded log file for more details.**