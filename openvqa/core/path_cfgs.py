# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):
        self.init_path()
        # self.check_path()


    def init_path(self):

        self.DATA_ROOT = './data'
        # self.DATA_ROOT = '/data/datasets'
        # self.DATA_ROOT = '/data1/datasets'
        # self.DATA_ROOT = '/home/features'

        self.DATA_PATH = {
            'vqa': self.DATA_ROOT + '/vqa',
            'clevr': self.DATA_ROOT + '/clevr',
        }


        self.FEATS_PATH = {
            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/feats' + '/train2014',
                'val': self.DATA_PATH['vqa'] + '/feats' + '/val2014',
                # 'test': self.DATA_PATH['vqa'] + '/feats' + '/test2015',
            },
            'clevr': {
                'train': self.DATA_PATH['clevr'] + '/feats' + '/train',
                'val': self.DATA_PATH['clevr'] + '/feats' + '/val',
                'test': self.DATA_PATH['clevr'] + '/feats' + '/test',
            },
        }


        self.RAW_PATH = {
            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_train2014_questions.json',
                'train-anno': self.DATA_PATH['vqa'] + '/raw' + '/v2_mscoco_train2014_annotations.json',
                'val': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_val2014_questions.json',
                'val-anno': self.DATA_PATH['vqa'] + '/raw' + '/v2_mscoco_val2014_annotations.json',
                'vg': self.DATA_PATH['vqa'] + '/raw' + '/VG_questions.json',
                'vg-anno': self.DATA_PATH['vqa'] + '/raw' + '/VG_annotations.json',
                'test': self.DATA_PATH['vqa'] + '/raw' + '/v2_OpenEnded_mscoco_test2015_questions.json',
            },
            'clevr': {
                'train': self.DATA_PATH['clevr'] + '/raw' + '/questions/CLEVR_train_questions.json',
                'val': self.DATA_PATH['clevr'] + '/raw' + '/questions/CLEVR_val_questions.json',
                'test': self.DATA_PATH['clevr'] + '/raw' + '/questions/CLEVR_test_questions.json',
            },
        }


        self.SPLITS = {
            'vqa': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },
            'clevr': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },

        }


        self.RESULT_PATH = './results/result_test'
        self.PRED_PATH = './results/pred'
        self.CACHE_PATH = './results/cache'
        self.LOG_PATH = './results/log'
        self.CKPTS_PATH = './ckpts'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self, dataset=None):
        print('Checking dataset ........')


        if dataset:
            for item in self.FEATS_PATH[dataset]:
                if not os.path.exists(self.FEATS_PATH[dataset][item]):
                    print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)

            for item in self.RAW_PATH[dataset]:
                if not os.path.exists(self.RAW_PATH[dataset][item]):
                    print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)

        else:
            for dataset in self.FEATS_PATH:
                for item in self.FEATS_PATH[dataset]:
                    if not os.path.exists(self.FEATS_PATH[dataset][item]):
                        print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

            for dataset in self.RAW_PATH:
                for item in self.RAW_PATH[dataset]:
                    if not os.path.exists(self.RAW_PATH[dataset][item]):
                        print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

        print('Finished!')
        print('')

