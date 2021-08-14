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

**Note that the `BINARIZE` will not influence the training, it only changes the evaluate behavior in TRAR, so the `BINARIZE=True` and `BINARIZE=False` share the same weight.**

| Model    | Base lr | ORDERS      | ROUTING    | ROUTING_MODE| POLICY    | BINARIZE |Overall (%) | Yes/No (%) | Number (%) | Other (%) | Download             |
|:--------:|:-------:|:-----------:|:----------:|:-----------:|:---------:|:--------:|:----------:|:----------:|:----------:|:---------:|:-------------------: |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| hard       | attention   | 1         | False    |  67.61     | 85.22      | 49.66      | 58.97     | [model](https://1drv.ms/f/s!Ary9y5k2nMUxhUNnPf0VnhX-eDW5) \| [log](https://1drv.ms/f/s!Ary9y5k2nMUxhUNnPf0VnhX-eDW5) |
| **TRAR** | 1e-4    | [0, 1, 2, 3]| hard       | attention   | 1         | True     |  67.62     | 85.19      | 49.75      | 58.98     |           -          |
