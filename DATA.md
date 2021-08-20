## Dataset Setup
you should prepare the following datasets before running the experiments

**if you only want to run experiments on one specific dataset, you can focus on the setup for the specific task**

### VQA-v2
- Image Features

We use `Grid-Features` extracted by the pretrained `ResNext152` model based on [grid-feats-vqa](), with each image being represented as an dynamic number (maximum number equals to 608) of 2048-D features. We first padded each feature into `32 × 32` scale and then pooled it by a kernel size of `2 × 2` with a stride of `2` to get our `16 × 16` feature. We did the same pooling operation to get the smaller feature with scale of `8 × 8`. We save the features for each image as a `.npy` file. **We only provide our extracted `8 × 8` features here, you can download the extracted features from [OneDrive](https://1drv.ms/f/s!Ary9y5k2nMUxhVGP9crDwW-97LrF) or [BaiduYun](https://pan.baidu.com/s/1GJL_yn6rJGFXypVbNR5e-g) with code `igr6`** The downloaded files containes three files: **train2014.zip, val2014.zip, and test2015.zip**, corresponding to the features of the `train/val/test` images for VQA-v2, respectively.

All the image features file should be unzipped to `data/vqa/feats` folder as the following data structure:

```
|-- data
	|-- vqa
	|  |-- feats
	|  |  |-- train2014
	|  |  |  |-- COCO_train2014_...jpg.npy
	|  |  |  |-- ...
	|  |  |-- val2014
	|  |  |  |-- COCO_val2014_...jpg.npy
	|  |  |  |-- ...
	|  |  |-- test2015
	|  |  |  |-- COCO_test2015_...jpg.npy
	|  |  |  |-- ...
```

**Extract Feature By Yourself**

If you want to train TRAR on `16 × 16` features, you can extract the features by yourself following these steps: 

1. clone **our own extension** of `grid-feats-vqa` repo:
```bash
$ git clone https://github.com/rentainhe/TRAR-Feature-Extraction.git
```
2. check the following tutorial [TRAR_Feature_Extraction](https://github.com/rentainhe/TRAR-Feature-Extraction/blob/master/TRAR_FEATURE_EXTRACTION.md) for more details. 

- QA Annotations

Download all the annotation `json` file for VQA-v2 from [OneDrive](https://onedrive.live.com/?id=31C59C3699CBBDBC%21702&cid=31C59C3699CBBDBC) or [BaiduYun](https://pan.baidu.com/s/14wUWg23wuanj7mF_BiGrIQ) with code `6fb6`

In addition, You can also use VQA samples from the Visual Genome to augment the training samples following [openvqa](https://github.com/MILVLG/openvqa), you can directly download VG question and annotation files from [OneDrive](https://awma1-my.sharepoint.com/personal/yuz_l0_tn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuz%5Fl0%5Ftn%2FDocuments%2Fshare%2Fvisualgenome%5Fqa&originalPath=aHR0cHM6Ly9hd21hMS1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC95dXpfbDBfdG4vRW1WSFZlR2RjazFJaWZQY3pHbVhvYU1CRmlTdnNlZ0E2dGZfUHF4TDNIWGNsdz9ydGltZT1SUU1pT3hwZTJVZw) or [BaiduYun](https://pan.baidu.com/s/1QCOtSxJGQA01DnhUg7FFtQ#list/path=%2F)

All the QA annotation files should be unzipped to `data/vqa/raw` folder as the following data structure:
```
|-- data
	|-- vqa
	|  |-- raw
	|  |  |-- v2_OpenEnded_mscoco_train2014_questions.json
	|  |  |-- v2_OpenEnded_mscoco_val2014_questions.json
	|  |  |-- v2_OpenEnded_mscoco_test2015_questions.json
	|  |  |-- v2_OpenEnded_mscoco_test-dev2015_questions.json
	|  |  |-- v2_mscoco_train2014_annotations.json
	|  |  |-- v2_mscoco_val2014_annotations.json
	|  |  |-- VG_questions.json
	|  |  |-- VG_annotations.json
```

### CLEVR
We built CLEVR dataset following [openvqa](https://github.com/MILVLG/openvqa)

- Images, Questions and Scene Graphs

Download all the [CLEVR v1.0]() from the official site, including all the splits needed for training, validation and testing.

All the image files, question files and scene graphs should be unzipped to `data/clevr/raw` folder as the following data structure:
```
|-- data
	|-- clevr
	|  |-- raw
	|  |  |-- images
	|  |  |  |-- train
	|  |  |  |  |-- CLEVR_train_000000.json
	|  |  |  |  |-- ...
	|  |  |  |  |-- CLEVR_train_069999.json
	|  |  |  |-- val
	|  |  |  |  |-- CLEVR_val_000000.json
	|  |  |  |  |-- ...
	|  |  |  |  |-- CLEVR_val_014999.json
	|  |  |  |-- test
	|  |  |  |  |-- CLEVR_test_000000.json
	|  |  |  |  |-- ...
	|  |  |  |  |-- CLEVR_test_014999.json
	|  |  |-- questions
	|  |  |  |-- CLEVR_train_questions.json
	|  |  |  |-- CLEVR_val_questions.json
	|  |  |  |-- CLEVR_test_questions.json
	|  |  |-- scenes
	|  |  |  |-- CLEVR_train_scenes.json
	|  |  |  |-- CLEVR_val_scenes.json
```

- Image Features

Following the previous work, we extract iamge features using a pretrained ResNet-101 model and generate `.h5` files, with each file corresponding to one image.
```bash
$ cd data/clevr
$ python clevr_extract_feat.py --mode=all --gpu=0
```

All the processed feature files should be placed in `data/clevr/feats` folder as the following data structrue:
```
|-- data
	|-- clevr
	|  |-- feats
	|  |  |-- train
	|  |  |  |-- 1.npz
	|  |  |  |-- ...
	|  |  |-- val
	|  |  |  |-- 1.npz
	|  |  |  |-- ...
	|  |  |-- test
	|  |  |  |-- 1.npz
	|  |  |  |-- ...
```


### FQAs
**Q:** When running `clevr_extract_feat.py` comes up `ImportError: cannot import name 'imread'`

**A:** Make sure you have already install `Pillow` first. If it still not work, you should use a lower version of `scipy`.
```bash
$ pip install Pillow
$ pip install scipy==1.2.1
```