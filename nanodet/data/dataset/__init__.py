# encoding=utf-8

import copy
from .coco import CocoDataset, MyDataset


def build_dataset(cfg, mode):
    """
    :param cfg:
    :param mode:
    :return:
    """
    dataset_cfg = copy.deepcopy(cfg)

    if dataset_cfg['name'] == 'coco':
        dataset_cfg.pop('name')
        return CocoDataset(mode=mode, **dataset_cfg)
    elif dataset_cfg['name'] == 'mcmot_det':
        dataset_cfg.pop('name')
        return MyDataset(mode=mode, **dataset_cfg)
